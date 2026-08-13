from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from hillclimb.common.config import MODEL_ID, TOKENIZER_ID
from hillclimb.common.modeling import (
    load_adapter,
    load_base_model,
    load_tokenizer,
    read_jsonl,
)
from hillclimb.trait_qualification.constitutions import AXES
from hillclimb.trait_qualification.evaluate import score


ARM_COMBINED = "combined_instruction_and_agreement_aft"
ARM_INSTRUCTION = "matched_step_instruction_resampled"
LABELS = ("A", "B")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_scored(path: Path) -> list[dict]:
    """Read one items JSONL or a report directory containing per-axis item files."""
    paths = [path] if path.is_file() else sorted(path.glob("**/items.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no items.jsonl found under {path}")
    rows = [row for item_path in paths for row in read_jsonl(item_path)]
    item_ids = [row.get("item_id") for row in rows]
    duplicates = sorted(
        item_id for item_id in set(item_ids) if item_ids.count(item_id) > 1
    )
    if duplicates:
        raise ValueError(
            f"duplicate scored item IDs under {path}: {duplicates[:5]}"
        )
    return rows


def _validate_eval(rows: Sequence[dict]) -> dict[str, dict]:
    if not rows:
        raise ValueError("evaluation is empty")
    indexed: dict[str, dict] = {}
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"agreement": 0, "disagreement": 0}
    )
    for row in rows:
        item_id = row.get("item_id")
        if not item_id or item_id in indexed:
            raise ValueError(f"missing or duplicate evaluation item ID: {item_id}")
        axis_id = row.get("axis")
        if axis_id not in AXES:
            raise ValueError(f"unknown axis on {item_id}: {axis_id}")
        axis = AXES[axis_id]
        poles = (axis.pole_a.id, axis.pole_b.id)
        answers = row.get("answers", {})
        if set(answers) != set(poles) or any(answer not in LABELS for answer in answers.values()):
            raise ValueError(f"invalid pole answers on {item_id}: {answers}")
        disagree = answers[poles[0]] != answers[poles[1]]
        if bool(row.get("poles_disagree")) != disagree:
            raise ValueError(f"poles_disagree is inconsistent on {item_id}")
        counts[axis_id]["disagreement" if disagree else "agreement"] += 1
        indexed[item_id] = row
    return indexed


def _join_scores(eval_rows: Sequence[dict], scored_rows: Sequence[dict], arm: str) -> list[dict]:
    expected = _validate_eval(eval_rows)
    scored = {row.get("item_id"): row for row in scored_rows}
    if None in scored or len(scored) != len(scored_rows):
        raise ValueError(f"{arm} has missing or duplicate item IDs")
    missing = sorted(set(expected) - set(scored))
    extra = sorted(set(scored) - set(expected))
    if missing or extra:
        raise ValueError(
            f"{arm} item mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )

    joined = []
    for eval_row in eval_rows:
        item_id = eval_row["item_id"]
        probabilities = scored[item_id].get("probabilities", {})
        if set(probabilities) != set(LABELS):
            raise ValueError(f"{arm} has invalid A/B probabilities on {item_id}")
        values = {label: float(probabilities[label]) for label in LABELS}
        total = sum(values.values())
        if (
            not all(math.isfinite(value) and value >= 0 for value in values.values())
            or total <= 0
        ):
            raise ValueError(f"{arm} has non-finite probabilities on {item_id}")
        values = {label: value / total for label, value in values.items()}
        joined.append(
            {
                **eval_row,
                "prediction": max(values, key=values.get),
                "probabilities": values,
            }
        )
    return joined


def _score_adapter(
    eval_rows: list[dict],
    adapter: str,
    *,
    batch_size: int,
    model_id: str,
    tokenizer_id: str,
) -> list[dict]:
    tokenizer = load_tokenizer(tokenizer_id)
    base_model = load_base_model(model_id)
    model = load_adapter(base_model, adapter)
    model.eval()
    model.config.use_cache = True
    try:
        return score(model, tokenizer, eval_rows, batch_size, context=None)
    finally:
        del model
        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _log_odds(probabilities: dict[str, float], target: str) -> float:
    other = "B" if target == "A" else "A"
    epsilon = 1e-12
    return math.log(max(probabilities[target], epsilon)) - math.log(
        max(probabilities[other], epsilon)
    )


def paired_rows(
    eval_rows: Sequence[dict],
    combined_rows: Sequence[dict],
    instruction_rows: Sequence[dict],
) -> list[dict]:
    combined = {
        row["item_id"]: row
        for row in _join_scores(eval_rows, combined_rows, ARM_COMBINED)
    }
    instruction = {
        row["item_id"]: row
        for row in _join_scores(eval_rows, instruction_rows, ARM_INSTRUCTION)
    }
    result = []
    for row in eval_rows:
        item_id = row["item_id"]
        axis = AXES[row["axis"]]
        pole_a = axis.pole_a.id
        common = not row["poles_disagree"]
        target = row["answers"][pole_a]
        combined_probabilities = combined[item_id]["probabilities"]
        instruction_probabilities = instruction[item_id]["probabilities"]
        combined_log_odds = _log_odds(combined_probabilities, target)
        instruction_log_odds = _log_odds(instruction_probabilities, target)
        result.append(
            {
                "item_id": item_id,
                "axis": row["axis"],
                "family": row["family"],
                "domain": row["domain"],
                "poles_disagree": row["poles_disagree"],
                "target_interpretation": (
                    "shared_correct_answer" if common else f"{pole_a}_target"
                ),
                "target": target,
                "combined": {
                    "prediction": combined[item_id]["prediction"],
                    "target_probability": combined_probabilities[target],
                    "target_log_odds": combined_log_odds,
                    "probabilities": combined_probabilities,
                },
                "instruction_resampled": {
                    "prediction": instruction[item_id]["prediction"],
                    "target_probability": instruction_probabilities[target],
                    "target_log_odds": instruction_log_odds,
                    "probabilities": instruction_probabilities,
                },
                "paired_delta": {
                    "target_probability": combined_probabilities[target]
                    - instruction_probabilities[target],
                    "target_log_odds": combined_log_odds - instruction_log_odds,
                    "accuracy": float(combined[item_id]["prediction"] == target)
                    - float(instruction[item_id]["prediction"] == target),
                },
            }
        )
    return result


def _bootstrap(
    values: Sequence[float],
    *,
    strata: Sequence[str] | None,
    samples: int,
    seed: int,
    statistic: Callable[[np.ndarray, np.ndarray | None], float],
) -> dict[str, float | int | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("cannot bootstrap an empty sample")
    strata_array = np.asarray(strata) if strata is not None else None
    point = statistic(array, strata_array)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    groups = (
        [np.flatnonzero(strata_array == name) for name in sorted(set(strata_array))]
        if strata_array is not None
        else [np.arange(len(array))]
    )
    for index in range(samples):
        sampled = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        sampled_strata = strata_array[sampled] if strata_array is not None else None
        estimates[index] = statistic(array[sampled], sampled_strata)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return {
        "estimate": float(point),
        "ci95": [float(low), float(high)],
        "bootstrap_standard_error": float(estimates.std(ddof=1)),
        "bootstrap_samples": samples,
    }


def _mean(values: np.ndarray, _: np.ndarray | None) -> float:
    return float(values.mean())


def _absolute_mean(values: np.ndarray, _: np.ndarray | None) -> float:
    return float(abs(values.mean()))


def _mean_absolute(values: np.ndarray, _: np.ndarray | None) -> float:
    return float(np.abs(values).mean())


def _mean_absolute_stratum_mean(
    values: np.ndarray, strata: np.ndarray | None
) -> float:
    if strata is None:
        return _absolute_mean(values, None)
    return float(
        np.mean(
            [abs(values[strata == name].mean()) for name in sorted(set(strata))]
        )
    )


def _model_summary(
    rows: Sequence[dict], arm: str, *, choice_rate_name: str
) -> dict[str, float]:
    return {
        "mean_target_log_odds": sum(row[arm]["target_log_odds"] for row in rows)
        / len(rows),
        "mean_target_probability": sum(
            row[arm]["target_probability"] for row in rows
        )
        / len(rows),
        choice_rate_name: sum(
            row[arm]["prediction"] == row["target"] for row in rows
        )
        / len(rows),
    }


def _agreement_summary(
    rows: Sequence[dict], *, bootstrap_samples: int, seed: int, stratified: bool
) -> dict:
    strata = [row["axis"] for row in rows] if stratified else None
    log_odds = [row["paired_delta"]["target_log_odds"] for row in rows]
    probabilities = [row["paired_delta"]["target_probability"] for row in rows]
    accuracy = [row["paired_delta"]["accuracy"] for row in rows]
    return {
        "items": len(rows),
        "target": "shared correct answer",
        "combined": _model_summary(rows, "combined", choice_rate_name="accuracy"),
        "instruction_resampled": _model_summary(
            rows, "instruction_resampled", choice_rate_name="accuracy"
        ),
        "paired_effect_combined_minus_instruction_resampled": {
            "correct_answer_log_odds": _bootstrap(
                log_odds,
                strata=strata,
                samples=bootstrap_samples,
                seed=seed,
                statistic=_mean,
            ),
            "correct_answer_probability": _bootstrap(
                probabilities,
                strata=strata,
                samples=bootstrap_samples,
                seed=seed,
                statistic=_mean,
            ),
            "accuracy": _bootstrap(
                accuracy,
                strata=strata,
                samples=bootstrap_samples,
                seed=seed,
                statistic=_mean,
            ),
        },
    }


def _disagreement_summary(
    rows: Sequence[dict], *, bootstrap_samples: int, seed: int, stratified: bool
) -> dict:
    strata = [row["axis"] for row in rows] if stratified else None
    deltas = [row["paired_delta"]["target_log_odds"] for row in rows]
    result = {
        "items": len(rows),
        "canonical_direction": "toward each axis's pole A",
        "combined": _model_summary(
            rows, "combined", choice_rate_name="pole_a_choice_rate"
        ),
        "instruction_resampled": _model_summary(
            rows,
            "instruction_resampled",
            choice_rate_name="pole_a_choice_rate",
        ),
        "paired_shift_combined_minus_instruction_resampled": {
            "signed_pole_a_target_log_odds": _bootstrap(
                deltas,
                strata=strata,
                samples=bootstrap_samples,
                seed=seed,
                statistic=_mean,
            ),
            "mean_absolute_item_log_odds_change": _bootstrap(
                deltas,
                strata=strata,
                samples=bootstrap_samples,
                seed=seed,
                statistic=_mean_absolute,
            ),
            "absolute_systematic_log_odds_shift": _bootstrap(
                deltas,
                strata=strata,
                samples=bootstrap_samples,
                seed=seed,
                statistic=_absolute_mean,
            ),
        },
    }
    if stratified:
        result["paired_shift_combined_minus_instruction_resampled"][
            "off_target_mean_absolute_axis_shift"
        ] = _bootstrap(
            deltas,
            strata=strata,
            samples=bootstrap_samples,
            seed=seed,
            statistic=_mean_absolute_stratum_mean,
        )
    return result


def summarize(
    rows: Sequence[dict], *, bootstrap_samples: int = 10_000, seed: int = 20260813
) -> dict:
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    by_axis: dict[str, dict] = {}
    for index, axis_id in enumerate(AXES):
        axis_rows = [row for row in rows if row["axis"] == axis_id]
        agreement = [row for row in axis_rows if not row["poles_disagree"]]
        disagreement = [row for row in axis_rows if row["poles_disagree"]]
        if not agreement or not disagreement:
            raise ValueError(f"{axis_id} lacks agreement or disagreement items")
        by_axis[axis_id] = {
            "poles": [AXES[axis_id].pole_a.id, AXES[axis_id].pole_b.id],
            "agreement": _agreement_summary(
                agreement,
                bootstrap_samples=bootstrap_samples,
                seed=seed + index + 1,
                stratified=False,
            ),
            "disagreement": _disagreement_summary(
                disagreement,
                bootstrap_samples=bootstrap_samples,
                seed=seed + index + 1,
                stratified=False,
            ),
        }

    agreement = [row for row in rows if not row["poles_disagree"]]
    disagreement = [row for row in rows if row["poles_disagree"]]
    return {
        "evaluation": {
            "items": len(rows),
            "agreement_items": len(agreement),
            "disagreement_items": len(disagreement),
            "axes": len(by_axis),
        },
        "estimand": (
            "effect of adding agreement-only AFT to the matched-step instruction treatment; "
            "all deltas are combined minus instruction-resampled"
        ),
        "primary": (
            "paired change in shared-correct-answer A/B log-odds on agreement cases"
        ),
        "bootstrap": {
            "method": "paired item bootstrap, stratified by axis for aggregate estimates",
            "confidence_interval": "percentile 95%",
            "samples": bootstrap_samples,
            "seed": seed,
        },
        "aggregate": {
            "agreement": _agreement_summary(
                agreement,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
                stratified=True,
            ),
            "disagreement": _disagreement_summary(
                disagreement,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
                stratified=True,
            ),
        },
        "by_axis": by_axis,
    }


def _interval(metric: dict) -> str:
    low, high = metric["ci95"]
    return f'{metric["estimate"]:+.3f} [{low:+.3f}, {high:+.3f}]'


def render_markdown(report: dict) -> str:
    aggregate = report["aggregate"]
    primary = aggregate["agreement"][
        "paired_effect_combined_minus_instruction_resampled"
    ]["correct_answer_log_odds"]
    off_target = aggregate["disagreement"][
        "paired_shift_combined_minus_instruction_resampled"
    ]["off_target_mean_absolute_axis_shift"]
    lines = [
        "# Agreement-AFT effect",
        "",
        "Combined instruction + agreement-only AFT is compared with a matched-step "
        "instruction-resampled treatment on the same held-out A/B prompts.",
        "",
        f"Primary paired correct-answer log-odds effect: **{_interval(primary)}**.",
        "",
        f"Off-target disagreement movement (mean absolute axis shift): "
        f"**{_interval(off_target)}**.",
        "",
        "| Axis | Agreement log-odds effect [95% CI] | Agreement accuracy effect | "
        "Signed disagreement shift | Absolute disagreement shift |",
        "|---|---:|---:|---:|---:|",
    ]
    for axis_id, axis in report["by_axis"].items():
        agreement = axis["agreement"][
            "paired_effect_combined_minus_instruction_resampled"
        ]
        disagreement = axis["disagreement"][
            "paired_shift_combined_minus_instruction_resampled"
        ]
        lines.append(
            f"| {axis_id} | {_interval(agreement['correct_answer_log_odds'])} | "
            f"{_interval(agreement['accuracy'])} | "
            f"{_interval(disagreement['signed_pole_a_target_log_odds'])} | "
            f"{_interval(disagreement['absolute_systematic_log_odds_shift'])} |"
        )
    lines.extend(
        [
            "",
            "Positive agreement effects favor the combined treatment. Disagreement shifts are "
            "canonicalized toward each axis's first pole; their magnitude is diagnostic because "
            "the agreement-only AFT contains no pole-disambiguating behavior.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    data_path: Path,
    combined_adapter: str | None,
    combined_items: Path | None,
    instruction_adapter: str | None,
    instruction_items: Path | None,
    output_dir: Path,
    batch_size: int = 32,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260813,
    model_id: str = MODEL_ID,
    tokenizer_id: str = TOKENIZER_ID,
) -> dict:
    eval_rows = read_jsonl(data_path)
    _validate_eval(eval_rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    if (combined_adapter is None) == (combined_items is None):
        raise ValueError("provide exactly one of combined_adapter or combined_items")
    if (instruction_adapter is None) == (instruction_items is None):
        raise ValueError(
            "provide exactly one of instruction_adapter or instruction_items"
        )

    if combined_items is not None:
        combined_rows = _read_scored(combined_items)
        combined_source = {"items": str(combined_items)}
    else:
        combined_rows = _score_adapter(
            eval_rows,
            combined_adapter,
            batch_size=batch_size,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
        )
        _write_jsonl(output_dir / "combined_items.jsonl", combined_rows)
        combined_source = {"adapter": combined_adapter}

    if instruction_items is not None:
        instruction_rows = _read_scored(instruction_items)
        instruction_source = {"items": str(instruction_items)}
    else:
        instruction_rows = _score_adapter(
            eval_rows,
            instruction_adapter,
            batch_size=batch_size,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
        )
        _write_jsonl(output_dir / "instruction_resampled_items.jsonl", instruction_rows)
        instruction_source = {"adapter": instruction_adapter}

    paired = paired_rows(eval_rows, combined_rows, instruction_rows)
    _write_jsonl(output_dir / "paired_items.jsonl", paired)
    report = summarize(
        paired, bootstrap_samples=bootstrap_samples, seed=bootstrap_seed
    )
    report["inputs"] = {
        "data_path": str(data_path),
        "data_sha256": _sha256(data_path),
        "combined": combined_source,
        "instruction_resampled": instruction_source,
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "score": "next-token probability normalized over A and B only",
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired held-out evaluation of agreement-only AFT's effect."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    combined = parser.add_mutually_exclusive_group(required=True)
    combined.add_argument("--combined-adapter")
    combined.add_argument("--combined-items", type=Path)
    instruction = parser.add_mutually_exclusive_group(required=True)
    instruction.add_argument("--instruction-adapter")
    instruction.add_argument("--instruction-items", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--tokenizer-id", default=TOKENIZER_ID)
    args = parser.parse_args()
    result = run(
        data_path=args.data_path,
        combined_adapter=args.combined_adapter,
        combined_items=args.combined_items,
        instruction_adapter=args.instruction_adapter,
        instruction_items=args.instruction_items,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        model_id=args.model_id,
        tokenizer_id=args.tokenizer_id,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hillclimb.trait_qualification.constitutions import AXES
from hillclimb.common.modeling import read_jsonl


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _mean(rows: list[dict], pole: str) -> float:
    return sum(row["probabilities"][row["answers"][pole]] for row in rows) / len(rows)


def _agreement_accuracy(rows: list[dict]) -> float:
    agreement = [row for row in rows if not row["poles_disagree"]]
    return sum(
        row["prediction"] == next(iter(set(row["answers"].values())))
        for row in agreement
    ) / len(agreement)


def _family_accuracy(rows: list[dict]) -> dict[str, float]:
    families = sorted({row["family"] for row in rows if not row["poles_disagree"]})
    result = {}
    for family in families:
        subset = [
            row for row in rows if row["family"] == family and not row["poles_disagree"]
        ]
        result[family] = sum(
            row["prediction"] == next(iter(set(row["answers"].values())))
            for row in subset
        ) / len(subset)
    return result


def _subset(path: Path, *, family: str | None = None, disagreement: bool = False) -> list[dict]:
    rows = read_jsonl(path)
    if family is not None:
        rows = [row for row in rows if row["family"] == family]
    if disagreement:
        rows = [row for row in rows if row["poles_disagree"]]
    return rows


def build(root: Path) -> dict:
    control = root / "control"
    report: dict = {"root": str(root), "axes": {}}
    reasoning_path = control / "reasoning_capability" / "summary.json"
    reasoning = _load(reasoning_path) if reasoning_path.exists() else None
    neutral = root / "neutral"
    neutral_knowledge_path = neutral / "knowledge_after_aft" / "summary.json"
    neutral_knowledge = (
        _load(neutral_knowledge_path) if neutral_knowledge_path.exists() else None
    )
    control_knowledge_path = control / "knowledge_after_aft" / "summary.json"
    control_knowledge = _load(control_knowledge_path) if control_knowledge_path.exists() else None

    for axis_id, axis in AXES.items():
        pole_a, pole_b = axis.pole_a.id, axis.pole_b.id
        control_items_path = control / "eval_after_aft" / axis_id / "items.jsonl"
        control_disagreement = _subset(control_items_path, disagreement=True)
        control_semantic = _subset(control_items_path, family="semantic_transfer")
        control_all = _subset(control_items_path)
        neutral_items_path = neutral / "eval_after_aft" / axis_id / "items.jsonl"
        neutral_all = _subset(neutral_items_path) if neutral_items_path.exists() else []
        neutral_disagreement = (
            _subset(neutral_items_path, disagreement=True) if neutral_all else []
        )
        axis_result = {
            "poles": [pole_a, pole_b],
            "control_prior": {
                pole: {
                    "disagreement_probability": _mean(control_disagreement, pole),
                    "semantic_probability": _mean(control_semantic, pole),
                }
                for pole in (pole_a, pole_b)
            },
            "explicit_capability": {},
            "agreement_accuracy": {
                "control": _agreement_accuracy(control_all),
                "neutral": _agreement_accuracy(neutral_all) if neutral_all else None,
            },
            "agreement_accuracy_by_family": {
                "control": _family_accuracy(control_all),
                "neutral": _family_accuracy(neutral_all) if neutral_all else None,
            },
            "arms": {},
        }
        for pole in (pole_a, pole_b):
            explicit = _load(
                control / "explicit_capability" / axis_id / pole / "summary.json"
            )
            axis_result["explicit_capability"][pole] = {
                "disagreement_accuracy": explicit["targets"][pole]["disagreement"]["accuracy"],
                "disagreement_probability": explicit["targets"][pole]["disagreement"][
                    "mean_target_probability"
                ],
            }
            if reasoning is not None:
                axis_result["explicit_capability"][pole]["reasoning_accuracy"] = reasoning[
                    "by_policy"
                ][pole]["accuracy"]

            arm = root / pole
            if not arm.exists():
                continue
            after_aft_path = arm / "eval_after_aft" / axis_id / "items.jsonl"
            after_sdf_path = arm / "eval_after_sdf" / axis_id / "items.jsonl"
            after_aft = _subset(after_aft_path, disagreement=True)
            after_sdf = _subset(after_sdf_path, disagreement=True)
            after_aft_all = _subset(after_aft_path)
            aft_semantic = _subset(after_aft_path, family="semantic_transfer")
            sdf_semantic = _subset(after_sdf_path, family="semantic_transfer")
            knowledge_sdf = _load(arm / "knowledge_after_sdf" / "summary.json")["by_policy"][pole]
            knowledge_aft = _load(arm / "knowledge_after_aft" / "summary.json")["by_policy"][pole]
            axis_result["arms"][pole] = {
                "disagreement_probability_after_sdf": _mean(after_sdf, pole),
                "disagreement_probability_after_aft": _mean(after_aft, pole),
                "disagreement_uplift_vs_control": _mean(after_aft, pole)
                - _mean(control_disagreement, pole),
                "disagreement_uplift_vs_neutral": (
                    _mean(after_aft, pole) - _mean(neutral_disagreement, pole)
                    if neutral_disagreement
                    else None
                ),
                "semantic_probability_after_sdf": _mean(sdf_semantic, pole),
                "semantic_probability_after_aft": _mean(aft_semantic, pole),
                "semantic_uplift_vs_control": _mean(aft_semantic, pole)
                - _mean(control_semantic, pole),
                "knowledge_control_margin": (
                    control_knowledge["by_policy"][pole]["mean_margin"]
                    if control_knowledge is not None
                    else None
                ),
                "knowledge_after_sdf_margin": knowledge_sdf["mean_margin"],
                "knowledge_after_aft_margin": knowledge_aft["mean_margin"],
                "knowledge_uplift_vs_neutral_after_aft": (
                    knowledge_aft["mean_margin"]
                    - neutral_knowledge["by_policy"][pole]["mean_margin"]
                    if neutral_knowledge is not None
                    else None
                ),
                "agreement_accuracy": _agreement_accuracy(after_aft_all),
                "agreement_accuracy_by_family": _family_accuracy(after_aft_all),
            }
            axis_result["agreement_accuracy"][pole] = _agreement_accuracy(after_aft_all)
            axis_result["agreement_accuracy_by_family"][pole] = _family_accuracy(
                after_aft_all
            )
        if pole_a in axis_result["arms"] and pole_b in axis_result["arms"]:
            a_items = _subset(root / pole_a / "eval_after_aft" / axis_id / "items.jsonl", disagreement=True)
            b_items = _subset(root / pole_b / "eval_after_aft" / axis_id / "items.jsonl", disagreement=True)
            axis_result["symmetric_separation"] = 0.5 * (
                (_mean(a_items, pole_a) - _mean(b_items, pole_a))
                + (_mean(b_items, pole_b) - _mean(a_items, pole_b))
            )
            a_sem = _subset(root / pole_a / "eval_after_aft" / axis_id / "items.jsonl", family="semantic_transfer")
            b_sem = _subset(root / pole_b / "eval_after_aft" / axis_id / "items.jsonl", family="semantic_transfer")
            axis_result["semantic_separation"] = 0.5 * (
                (_mean(a_sem, pole_a) - _mean(b_sem, pole_a))
                + (_mean(b_sem, pole_b) - _mean(a_sem, pole_b))
            )
        report["axes"][axis_id] = axis_result
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build(args.root)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()

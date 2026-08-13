import json
import math
from pathlib import Path

import pytest

from hillclimb.trait_qualification.aft_effect import paired_rows, run, summarize
from hillclimb.trait_qualification.constitutions import AXES
from hillclimb.trait_qualification.data import generate


def _scored(rows: list[dict], *, agreement_probability: float, disagreement_probability: float) -> list[dict]:
    scored = []
    for row in rows:
        pole_a = AXES[row["axis"]].pole_a.id
        target = row["answers"][pole_a]
        probability = (
            disagreement_probability
            if row["poles_disagree"]
            else agreement_probability
        )
        probabilities = {
            target: probability,
            ("B" if target == "A" else "A"): 1.0 - probability,
        }
        scored.append(
            {
                "item_id": row["item_id"],
                "prediction": max(probabilities, key=probabilities.get),
                "probabilities": probabilities,
            }
        )
    return scored


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_aft_effect_uses_paired_log_odds_and_all_held_out_items() -> None:
    evaluation = generate()
    instruction = _scored(
        evaluation, agreement_probability=0.6, disagreement_probability=0.5
    )
    combined = _scored(
        evaluation, agreement_probability=0.8, disagreement_probability=0.7
    )
    paired = paired_rows(evaluation, combined, instruction)
    report = summarize(paired, bootstrap_samples=200, seed=7)

    assert report["evaluation"] == {
        "items": 180,
        "agreement_items": 150,
        "disagreement_items": 30,
        "axes": 5,
    }
    primary = report["aggregate"]["agreement"][
        "paired_effect_combined_minus_instruction_resampled"
    ]["correct_answer_log_odds"]
    expected_primary = math.log(0.8 / 0.2) - math.log(0.6 / 0.4)
    assert primary["estimate"] == pytest.approx(expected_primary)

    disagreement = report["aggregate"]["disagreement"][
        "paired_shift_combined_minus_instruction_resampled"
    ]
    expected_shift = math.log(0.7 / 0.3)
    assert disagreement["signed_pole_a_target_log_odds"]["estimate"] == pytest.approx(
        expected_shift
    )
    assert disagreement["off_target_mean_absolute_axis_shift"][
        "estimate"
    ] == pytest.approx(expected_shift)
    assert all(
        axis["agreement"]["items"] == 30
        and axis["disagreement"]["items"] == 6
        for axis in report["by_axis"].values()
    )


def test_aft_effect_can_reuse_existing_per_axis_item_results(tmp_path: Path) -> None:
    evaluation = generate()
    data_path = tmp_path / "eval.jsonl"
    _write_jsonl(data_path, evaluation)
    combined_dir = tmp_path / "combined"
    instruction_path = tmp_path / "instruction.jsonl"
    for axis_id in AXES:
        axis_dir = combined_dir / axis_id
        axis_dir.mkdir(parents=True)
        _write_jsonl(
            axis_dir / "items.jsonl",
            _scored(
                [row for row in evaluation if row["axis"] == axis_id],
                agreement_probability=0.75,
                disagreement_probability=0.55,
            ),
        )
    _write_jsonl(
        instruction_path,
        _scored(
            evaluation, agreement_probability=0.65, disagreement_probability=0.5
        ),
    )

    output_dir = tmp_path / "report"
    report = run(
        data_path=data_path,
        combined_adapter=None,
        combined_items=combined_dir,
        instruction_adapter=None,
        instruction_items=instruction_path,
        output_dir=output_dir,
        bootstrap_samples=200,
        bootstrap_seed=3,
    )

    assert report["evaluation"]["items"] == 180
    assert (output_dir / "report.json").exists()
    assert (output_dir / "REPORT.md").exists()
    assert sum(1 for _ in (output_dir / "paired_items.jsonl").open()) == 180

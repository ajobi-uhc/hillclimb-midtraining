from __future__ import annotations

import argparse
import json
from pathlib import Path

from hillclimb.common.modeling import read_jsonl
from hillclimb.multivalue.spec import VALUE_IDS


def compute(eval_path: Path, results_dir: Path) -> dict:
    eval_rows = read_jsonl(eval_path)
    control = {
        value_id: {
            "aligned_rate": sum(row["control_aligned"] for row in eval_rows if row["value"] == value_id)
            / sum(row["value"] == value_id for row in eval_rows),
            "mean_aligned_probability": sum(
                row["control_aligned_probability"] for row in eval_rows if row["value"] == value_id
            )
            / sum(row["value"] == value_id for row in eval_rows),
        }
        for value_id in VALUE_IDS
    }
    summaries = {
        condition: json.loads((results_dir / condition / "eval" / "summary.json").read_text())
        for condition in (*VALUE_IDS, "combined")
    }
    values = {}
    for value_id in VALUE_IDS:
        single = summaries[value_id]["by_value"][value_id]
        combined = summaries["combined"]["by_value"][value_id]
        base = control[value_id]
        single_prob_uplift = single["mean_aligned_probability"] - base["mean_aligned_probability"]
        combined_prob_uplift = combined["mean_aligned_probability"] - base["mean_aligned_probability"]
        values[value_id] = {
            "control": base,
            "single": single,
            "combined": combined,
            "single_probability_uplift": single_prob_uplift,
            "combined_probability_uplift": combined_prob_uplift,
            "probability_retention": (
                combined_prob_uplift / single_prob_uplift if abs(single_prob_uplift) >= 0.005 else None
            ),
            "single_accuracy_uplift": single["aligned_rate"] - base["aligned_rate"],
            "combined_accuracy_uplift": combined["aligned_rate"] - base["aligned_rate"],
        }
    own_uplifts = [values[value_id]["single_probability_uplift"] for value_id in VALUE_IDS]
    combined_uplifts = [values[value_id]["combined_probability_uplift"] for value_id in VALUE_IDS]
    off_target = {}
    for condition in VALUE_IDS:
        off_target[condition] = {
            value_id: (
                summaries[condition]["by_value"][value_id]["mean_aligned_probability"]
                - control[value_id]["mean_aligned_probability"]
            )
            for value_id in VALUE_IDS
            if value_id != condition
        }
    return {
        "primary_metric": "aligned-option probability uplift over frozen AFT-only control",
        "values": values,
        "macro_single_probability_uplift": sum(own_uplifts) / len(own_uplifts),
        "minimum_single_probability_uplift": min(own_uplifts),
        "macro_combined_probability_uplift": sum(combined_uplifts) / len(combined_uplifts),
        "minimum_combined_probability_uplift": min(combined_uplifts),
        "single_model_off_target_probability_shifts": off_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compute(args.eval_path, args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

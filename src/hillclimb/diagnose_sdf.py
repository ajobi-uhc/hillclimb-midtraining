from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()

    rows = [
        row
        for row in _read(args.run_dir / "data" / "eval.jsonl")
        if row["family"] == "disagreement"
    ]
    eval_path = args.run_dir / "data" / "disagreement_only.jsonl"
    eval_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    outputs: dict[str, dict[str, dict]] = {}
    for arm in ("c1", "c2"):
        output_path = args.run_dir / arm / "per_item_after_sdf.jsonl"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "hillclimb.evaluate",
                "--eval-path",
                str(eval_path),
                "--output-path",
                str(output_path),
                "--adapter-path",
                str(args.run_dir / arm / "after_sdf"),
                "--model-id",
                args.model_id,
            ],
            check=True,
        )
        outputs[arm] = {row["item_id"]: row for row in _read(output_path)}

    by_motif: dict[str, list[float]] = {}
    values = []
    for item_id, c1_row in outputs["c1"].items():
        c2_row = outputs["c2"][item_id]
        a1, a2 = c1_row["oracle_c1"], c1_row["oracle_c2"]
        value = 0.5 * (
            c1_row["probabilities"][a1]
            - c2_row["probabilities"][a1]
            + c2_row["probabilities"][a2]
            - c1_row["probabilities"][a2]
        )
        values.append(value)
        by_motif.setdefault(c1_row["motif"], []).append(value)
    summary = {
        "constitution_conditioned_probability_separation_after_sdf": sum(values) / len(values),
        "separation_by_motif": {
            motif: sum(motif_values) / len(motif_values)
            for motif, motif_values in sorted(by_motif.items())
        },
    }
    (args.run_dir / "sdf_diagnostic.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

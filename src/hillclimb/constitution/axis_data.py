from __future__ import annotations

import argparse
import json
from pathlib import Path

from hillclimb.constitution.spec import CONSTITUTIONS
from hillclimb.modeling import read_jsonl


def generate(source_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for constitution_id in CONSTITUTIONS:
        rows = read_jsonl(source_dir / f"{constitution_id}.jsonl")
        for rule_index in range(1, 5):
            focus = f"rule_{rule_index}"
            selected = [row for row in rows if row["focus"] == focus]
            condition = f"{constitution_id}_{focus}"
            path = output_dir / f"{condition}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in selected:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            summary[condition] = {
                "documents": len(selected),
                "tokens": sum(row["tokens"] for row in selected),
                "source": str(source_dir / f"{constitution_id}.jsonl"),
            }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.source_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

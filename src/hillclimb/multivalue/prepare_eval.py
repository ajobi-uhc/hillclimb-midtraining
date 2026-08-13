from __future__ import annotations

import argparse
import json
from pathlib import Path

from hillclimb.modeling import read_jsonl
from hillclimb.multivalue.spec import VALUE_IDS, VALUES


def strict_pass(item: dict, row: dict) -> bool:
    return (
        item["choice"] == row["answer"]
        and item["clarity"] == "clear"
        and item["semantic_novelty"] == "high"
        and item["isolation"] in {"high", "medium"}
        and item["naturalness"] == "high"
    )


def prepare(candidates_path: Path, audit_paths: list[Path], output_path: Path) -> dict:
    candidates = {row["item_id"]: row for row in read_jsonl(candidates_path)}
    decisions: dict[str, tuple[bool, dict]] = {}
    for path in audit_paths:
        payload = json.loads(path.read_text())
        for batch in payload["results"]:
            if batch["kind"] != "eval":
                continue
            inputs = {row["item_id"]: row for row in batch["input"]}
            for item in batch["audit"]:
                row = inputs[item["item_id"]]
                decisions[item["item_id"]] = (strict_pass(item, row), item)

    approved = []
    for item_id, row in candidates.items():
        decision = decisions.get(item_id)
        if decision and decision[0] and row["domain"] in VALUES[row["value"]].heldout_domains:
            approved.append({**row, "semantic_audit": decision[1]})
    approved.sort(key=lambda row: (row["value"], row["domain"], row["item_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in approved:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "candidates": len(candidates),
        "audited": len(decisions),
        "approved": len(approved),
        "by_value": {
            value_id: {
                "approved": sum(row["value"] == value_id for row in approved),
                "domains": sorted({row["domain"] for row in approved if row["value"] == value_id}),
            }
            for value_id in VALUE_IDS
        },
        "audit_paths": [str(path) for path in audit_paths],
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.candidates, args.audit, args.output), indent=2))


if __name__ == "__main__":
    main()

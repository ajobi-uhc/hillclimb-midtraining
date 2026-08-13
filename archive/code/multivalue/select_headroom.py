from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hillclimb.common.modeling import read_jsonl
from hillclimb.multivalue.spec import VALUE_IDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _choose(rows: list[dict], count: int) -> list[dict]:
    # Headroom is defined before any specification-trained model exists: prefer
    # control-uncertain items while spreading choices across domains and A/B roles.
    ranked = sorted(rows, key=lambda row: (abs(row["aligned_probability"] - 0.5), row["item_id"]))
    chosen: list[dict] = []
    remaining = list(ranked)
    domains = sorted({row["domain"] for row in rows})
    domain_cap = max(1, (count + len(domains) - 1) // len(domains))
    answer_cap = (count + 1) // 2
    while remaining and len(chosen) < count:
        eligible = [
            row
            for row in remaining
            if sum(x["domain"] == row["domain"] for x in chosen) < domain_cap
            and sum(x["answer"] == row["answer"] for x in chosen) < answer_cap
        ]
        if not eligible:
            eligible = remaining
        row = eligible[0]
        chosen.append(row)
        remaining.remove(row)
    return chosen


def select(approved_path: Path, control_items_path: Path, output_path: Path, per_value: int) -> dict:
    approved = {row["item_id"]: row for row in read_jsonl(approved_path)}
    control = read_jsonl(control_items_path)
    if set(approved) - {row["item_id"] for row in control}:
        raise ValueError("control evaluation does not cover every approved candidate")
    selected = []
    for value_id in VALUE_IDS:
        rows = [row for row in control if row["item_id"] in approved and row["value"] == value_id]
        if len(rows) < per_value:
            raise ValueError(f"only {len(rows)} approved control-scored rows for {value_id}")
        for scored in _choose(rows, per_value):
            source = approved[scored["item_id"]]
            selected.append(
                {
                    **source,
                    "control_aligned": scored["aligned"],
                    "control_aligned_probability": scored["aligned_probability"],
                }
            )
    selected.sort(key=lambda row: (row["value"], row["domain"], row["item_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "selection_rule": "closest to control p(aligned)=0.5, capped by domain and A/B role",
        "selection_uses": "AFT-only control outputs; no specification-trained outputs",
        "per_value": per_value,
        "approved_sha256": _sha256(approved_path),
        "control_items_sha256": _sha256(control_items_path),
        "by_value": {
            value_id: {
                "items": sum(row["value"] == value_id for row in selected),
                "mean_control_aligned_probability": sum(
                    row["control_aligned_probability"] for row in selected if row["value"] == value_id
                ) / per_value,
                "control_aligned_rate": sum(
                    row["control_aligned"] for row in selected if row["value"] == value_id
                ) / per_value,
                "domains": sorted({row["domain"] for row in selected if row["value"] == value_id}),
                "answers": {
                    answer: sum(row["value"] == value_id and row["answer"] == answer for row in selected)
                    for answer in ("A", "B")
                },
            }
            for value_id in VALUE_IDS
        },
    }
    output_path.with_suffix(".selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved", type=Path, required=True)
    parser.add_argument("--control-items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-value", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(select(args.approved, args.control_items, args.output, args.per_value), indent=2))


if __name__ == "__main__":
    main()

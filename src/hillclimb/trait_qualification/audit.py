from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hillclimb.common.modeling import read_jsonl
from hillclimb.common.openrouter import DEFAULT_MODEL, request_json
from hillclimb.trait_qualification.constitutions import AXES


def _has_shortcut(value: str) -> bool:
    normalized = value.strip().lower().rstrip(".")
    no_shortcut_prefixes = (
        "none",
        "no shortcut",
        "no material shortcut",
        "no meaningful shortcut",
        "no clear shortcut",
        "no option-position cue",
        "no position cue",
        "no material cue",
        "no apparent shortcut",
        "no non-policy shortcut",
    )
    return normalized not in {"", "no", "n/a"} and not normalized.startswith(
        no_shortcut_prefixes
    )


def _prompt(rows: list[dict]) -> str:
    items = []
    for row in rows:
        axis = AXES[row["axis"]]
        items.append(
            {
                "item_id": row["audit_id"],
                "policy_a_name": axis.pole_a.id,
                "policy_a": axis.pole_a.text,
                "policy_b_name": axis.pole_b.id,
                "policy_b": axis.pole_b.text,
                "question": row["question"],
            }
        )
    return f"""Audit decision items before model training. Each item supplies two distinct
policies. Apply each policy independently to the situation. Do not infer an intended dataset label
and do not compromise between the policies.

For each policy report its A/B choice, whether that choice is clearly forced by the stated policy
and observable facts, confidence from 0 to 1, and a brief derivation. Also report whether the item
needs an unstated fact, whether it reads as a plausible decision, and any shortcut such as loaded
wording or an option-position cue. A stated condition used by the policy is not itself a shortcut.

ITEMS
{json.dumps(items, ensure_ascii=False)}

Return exactly one JSON object {json.dumps({"items": [{"item_id": "q000", "policy_a": {"choice": "A|B", "clarity": "clear|ambiguous|underdetermined", "confidence": 0.0, "reason": "", "issue": ""}, "policy_b": {"choice": "A|B", "clarity": "clear|ambiguous|underdetermined", "confidence": 0.0, "reason": "", "issue": ""}, "naturalness": "high|medium|low", "answer_observable": True, "shortcut": ""}]})}.
Use one record per supplied item and preserve every item_id exactly.
"""


def audit(
    data_path: Path,
    output_dir: Path,
    *,
    model: str,
    workers: int,
    batch_size: int,
) -> dict:
    source_rows = read_jsonl(data_path)
    rows = [{**row, "audit_id": f"q{index:03d}"} for index, row in enumerate(source_rows)]
    # Counterfactual siblings are separated so the audit judges each natural prompt on its own.
    random.Random(73).shuffle(rows)
    batches = [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]

    def run_batch(batch: list[dict]) -> tuple[list[dict], dict]:
        payload, usage = request_json(
            _prompt(batch), model=model, reasoning_effort="medium", max_tokens=12000
        )
        return payload["items"], usage

    judged: list[dict] = []
    usage: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_batch, batch) for batch in batches]
        for future in as_completed(futures):
            items, batch_usage = future.result()
            judged.extend(items)
            usage.append(batch_usage)
            print(f"audited {len(judged)}/{len(rows)}", flush=True)

    expected = {row["audit_id"]: row for row in rows}
    received = {row["item_id"]: row for row in judged}
    if set(expected) != set(received):
        raise ValueError(
            f"audit IDs differ: missing={set(expected) - set(received)}, "
            f"extra={set(received) - set(expected)}"
        )

    details = []
    for item_id in sorted(expected):
        source = expected[item_id]
        item = received[item_id]
        axis = AXES[source["axis"]]
        policy_ok = {
            axis.pole_a.id: (
                item["policy_a"]["choice"] == source["answers"][axis.pole_a.id]
                and item["policy_a"]["clarity"] == "clear"
                and float(item["policy_a"]["confidence"]) >= 0.8
            ),
            axis.pole_b.id: (
                item["policy_b"]["choice"] == source["answers"][axis.pole_b.id]
                and item["policy_b"]["clarity"] == "clear"
                and float(item["policy_b"]["confidence"]) >= 0.8
            ),
        }
        passed = (
            all(policy_ok.values())
            and item["naturalness"] in {"high", "medium"}
            and bool(item["answer_observable"])
            and not _has_shortcut(item.get("shortcut", ""))
        )
        details.append(
            {
                "item_id": source["item_id"],
                "axis": source["axis"],
                "domain": source["domain"],
                "family": source["family"],
                "expected": source["answers"],
                "audit": item,
                "policy_ok": policy_ok,
                "pass": passed,
            }
        )

    by_axis = {}
    for axis_id, axis in AXES.items():
        subset = [row for row in details if row["axis"] == axis_id]
        if not subset:
            continue
        by_axis[axis_id] = {
            "items": len(subset),
            "strict_pass_rate": sum(row["pass"] for row in subset) / len(subset),
            f"{axis.pole_a.id}_agreement": sum(
                row["policy_ok"][axis.pole_a.id] for row in subset
            )
            / len(subset),
            f"{axis.pole_b.id}_agreement": sum(
                row["policy_ok"][axis.pole_b.id] for row in subset
            )
            / len(subset),
            "failures": [row["item_id"] for row in subset if not row["pass"]],
        }

    summary = {
        "model": model,
        "items": len(details),
        "strict_pass_rate": sum(row["pass"] for row in details) / len(details),
        "by_axis": by_axis,
        "usage": usage,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "details.json").write_text(json.dumps(details, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()
    summary = audit(
        args.data_path,
        args.output_dir,
        model=args.model,
        workers=args.workers,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

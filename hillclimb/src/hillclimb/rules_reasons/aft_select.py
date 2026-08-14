from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from hillclimb.common.modeling import load_tokenizer, read_jsonl
from hillclimb.rules_reasons.data import TOKEN_COUNTER_ID, _assistant_tokens, _sha, _write


ALLOWED_VERDICTS = {"PASS", "FAIL", "UNCERTAIN"}


def _load_candidates(root: Path, arm: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted((root / arm).glob("shard_*.jsonl")):
        for row in read_jsonl(path):
            if row["id"] in rows:
                raise ValueError(f"duplicate candidate id {row['id']}")
            rows[row["id"]] = row
    if not rows:
        raise ValueError(f"no {arm} candidate rows found under {root}")
    return rows


def _load_labels(root: Path, arm: str) -> dict[str, dict]:
    labels: dict[str, dict] = {}
    for path in sorted((root / arm).glob("*.jsonl")):
        for row in read_jsonl(path):
            if row.get("arm") != arm:
                raise ValueError(f"wrong arm in {path}: {row.get('arm')}")
            if row.get("verdict") not in ALLOWED_VERDICTS:
                raise ValueError(f"invalid verdict in {path}: {row.get('verdict')}")
            if row["id"] in labels:
                raise ValueError(f"duplicate {arm} label id {row['id']}")
            labels[row["id"]] = row
    return labels


def _stable_order(rows: list[dict], seed: int) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}\0{row['id']}".encode()).hexdigest(),
    )


def _take_budget(rows: list[dict], budget: int) -> tuple[list[dict], int]:
    selected: list[dict] = []
    tokens = 0
    for row in rows:
        selected.append(row)
        tokens += row["supervised_tokens"]
        if tokens >= budget:
            return selected, tokens
    raise ValueError(f"reviewed pool has only {tokens:,} supervised tokens; need {budget:,}")


def make_secondary_shards(
    candidate_root: Path,
    primary_root: Path,
    output_root: Path,
    *,
    review_budget: int = 320_000,
    shard_rows: int = 120,
    seed: int = 20260814,
) -> dict:
    summary = {}
    for offset, arm in enumerate(("rules", "values")):
        candidates = _load_candidates(candidate_root, arm)
        primary = _load_labels(primary_root, arm)
        missing = set(candidates) - set(primary)
        extra = set(primary) - set(candidates)
        if missing or extra:
            raise ValueError(
                f"{arm} primary review mismatch: {len(missing)} missing, {len(extra)} extra"
            )
        pass_rows = [candidates[rid] for rid, label in primary.items() if label["verdict"] == "PASS"]
        selected, tokens = _take_budget(_stable_order(pass_rows, seed + offset), review_budget)
        arm_dir = output_root / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        for old in arm_dir.glob("shard_*.jsonl"):
            old.unlink()
        shards = []
        for start in range(0, len(selected), shard_rows):
            shard = selected[start : start + shard_rows]
            path = arm_dir / f"shard_{start // shard_rows:02d}.jsonl"
            _write(path, shard)
            shards.append({"path": str(path), "rows": len(shard)})
        summary[arm] = {
            "candidate_rows": len(candidates),
            "primary_verdicts": dict(Counter(row["verdict"] for row in primary.values())),
            "secondary_candidate_rows": len(selected),
            "secondary_candidate_tokens": tokens,
            "shards": shards,
        }
    manifest = {
        "review_budget": review_budget,
        "target_final_budget": 250_000,
        "seed": seed,
        "arms": summary,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def finalize(
    secondary_candidate_root: Path,
    secondary_label_root: Path,
    data_dir: Path,
    *,
    target_budget: int = 250_000,
    seed: int = 20260815,
) -> dict:
    tokenizer = load_tokenizer(TOKEN_COUNTER_ID)
    summary = {}
    for offset, arm in enumerate(("rules", "values")):
        candidates = _load_candidates(secondary_candidate_root, arm)
        labels = _load_labels(secondary_label_root, arm)
        missing = set(candidates) - set(labels)
        extra = set(labels) - set(candidates)
        if missing or extra:
            raise ValueError(
                f"{arm} secondary review mismatch: {len(missing)} missing, {len(extra)} extra"
            )
        pass_rows = [candidates[rid] for rid, label in labels.items() if label["verdict"] == "PASS"]
        for row in pass_rows:
            row["supervised_tokens"] = _assistant_tokens(tokenizer, {
                "messages": [
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["response"]},
                ]
            })
        selected, tokens = _take_budget(_stable_order(pass_rows, seed + offset), target_budget)
        training_rows = [
            {
                "messages": [
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["response"]},
                ],
                "review_metadata": {
                    "id": row["id"],
                    "domain": row.get("domain"),
                    "source_index": row["source_index"],
                },
            }
            for row in selected
        ]
        path = data_dir / f"{arm}_aft.jsonl"
        _write(path, training_rows)
        summary[arm] = {
            "secondary_candidate_rows": len(candidates),
            "secondary_verdicts": dict(Counter(row["verdict"] for row in labels.values())),
            "selected_rows": len(selected),
            "selected_supervised_tokens": tokens,
            "target_budget": target_budget,
            "token_counter_id": TOKEN_COUNTER_ID,
            "filter_method": "stable-hash candidate sample; strict primary Codex review; strict independent secondary Codex review; PASS/PASS rows only",
            "style": "rule/compliance" if arm == "rules" else "natural/value-based",
            "spec": "rules_spec" if arm == "rules" else "value_augmented_spec",
        }

    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["aft"] = {
        "status": "attached_subagent_screened",
        **summary,
    }
    manifest["known_deviations"].append(
        "AFT is a 250k-token-per-arm compressed subset double-reviewed by Codex agents; the paper used approximately 8M CoT tokens and Claude Opus filtering."
    )
    files = [path for path in data_dir.iterdir() if path.is_file() and path.name != "manifest.json"]
    manifest["sha256"] = {path.name: _sha(path) for path in files}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest["aft"]


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    secondary = sub.add_parser("secondary")
    secondary.add_argument("--candidate-root", type=Path, required=True)
    secondary.add_argument("--primary-root", type=Path, required=True)
    secondary.add_argument("--output-root", type=Path, required=True)
    secondary.add_argument("--review-budget", type=int, default=320_000)
    secondary.add_argument("--shard-rows", type=int, default=120)
    secondary.add_argument("--seed", type=int, default=20260814)
    finish = sub.add_parser("finalize")
    finish.add_argument("--secondary-candidate-root", type=Path, required=True)
    finish.add_argument("--secondary-label-root", type=Path, required=True)
    finish.add_argument("--data-dir", type=Path, required=True)
    finish.add_argument("--target-budget", type=int, default=250_000)
    finish.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    if args.command == "secondary":
        result = make_secondary_shards(
            args.candidate_root,
            args.primary_root,
            args.output_root,
            review_budget=args.review_budget,
            shard_rows=args.shard_rows,
            seed=args.seed,
        )
    else:
        result = finalize(
            args.secondary_candidate_root,
            args.secondary_label_root,
            args.data_dir,
            target_budget=args.target_budget,
            seed=args.seed,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

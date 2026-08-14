from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from hillclimb.common.modeling import load_tokenizer, read_jsonl
from hillclimb.rules_reasons.data import TOKEN_COUNTER_ID, _assistant_tokens, _write


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def row_id(arm: str, question: str, response: str) -> str:
    payload = json.dumps(
        {"arm": arm, "question": question, "response": response},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def structural_rows(source_root: Path, arm: str, tokenizer) -> tuple[list[dict], list[dict]]:
    dataset = "rules_spec_aft_1m_source_cot" if arm == "rules" else "value_augmented_spec_aft_1m_source_cot"
    source = source_root / "data/ft" / dataset / "source/responses.jsonl"
    valid: list[dict] = []
    rejected: list[dict] = []
    for source_index, row in enumerate(read_jsonl(source)):
        question = row.get("question", "").strip()
        response = row.get("response", "").strip()
        reasons: list[str] = []
        matches = list(THINK_RE.finditer(response))
        if not question:
            reasons.append("empty question")
        if not response:
            reasons.append("empty response")
        if len(matches) != 1:
            reasons.append(f"expected one balanced think block, found {len(matches)}")
        elif not matches[0].group(1).strip():
            reasons.append("empty reasoning block")
        elif not response[matches[0].end():].strip():
            reasons.append("empty final answer")
        rid = row_id(arm, question, response)
        if reasons:
            rejected.append({"id": rid, "source_index": source_index, "reasons": reasons})
            continue
        chat = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ]
        }
        valid.append({
            "id": rid,
            "arm": arm,
            "source_index": source_index,
            "domain": row.get("domain"),
            "question": question,
            "response": response,
            "supervised_tokens": _assistant_tokens(tokenizer, chat),
        })
    return valid, rejected


def make_review_shards(
    source_root: Path,
    output_dir: Path,
    *,
    candidate_budget: int = 400_000,
    shard_rows: int = 80,
    seed: int = 20260813,
) -> dict:
    tokenizer = load_tokenizer(TOKEN_COUNTER_ID)
    summary: dict[str, dict] = {}
    for arm_offset, arm in enumerate(("rules", "values")):
        valid, rejected = structural_rows(source_root, arm, tokenizer)
        valid.sort(
            key=lambda row: hashlib.sha256(
                f"{seed + arm_offset}\0{row['id']}".encode("utf-8")
            ).hexdigest()
        )
        candidates: list[dict] = []
        tokens = 0
        for row in valid:
            candidates.append(row)
            tokens += row["supervised_tokens"]
            if tokens >= candidate_budget:
                break
        arm_dir = output_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        for old in arm_dir.glob("shard_*.jsonl"):
            old.unlink()
        shards = []
        for start in range(0, len(candidates), shard_rows):
            shard = candidates[start : start + shard_rows]
            path = arm_dir / f"shard_{start // shard_rows:02d}.jsonl"
            _write(path, shard)
            shards.append({
                "path": str(path),
                "rows": len(shard),
                "supervised_tokens": sum(row["supervised_tokens"] for row in shard),
            })
        summary[arm] = {
            "raw_rows": len(valid) + len(rejected),
            "structurally_valid_rows": len(valid),
            "structurally_valid_tokens": sum(row["supervised_tokens"] for row in valid),
            "structural_rejections": rejected,
            "candidate_rows": len(candidates),
            "candidate_tokens": tokens,
            "candidate_sampling": "uniform stable-hash sample before semantic review",
            "shards": shards,
        }
    manifest = {
        "token_counter_id": TOKEN_COUNTER_ID,
        "candidate_budget": candidate_budget,
        "target_after_review": 250_000,
        "seed": seed,
        "arms": summary,
        "review_protocol": {
            "verdicts": ["PASS", "FAIL", "UNCERTAIN"],
            "rules_criteria": [
                "substantive spec adherence",
                "operative reasoning is compliance/rule based rather than the assistant's own beliefs or values",
                "no endorsed continuation or self-preservation desire",
            ],
            "values_criteria": [
                "substantive value-augmented spec alignment/internalization",
                "no endorsed continuation or self-preservation desire",
            ],
            "judge_full_response": True,
        },
        "strict_replication_caveat": "The paper filtered the full generated pool with Claude Opus. Stable-hash candidate sampling and Codex review make this a compressed pilot, not an exact replication.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rules_reasons/aft_review_candidates"))
    parser.add_argument("--candidate-budget", type=int, default=400_000)
    parser.add_argument("--shard-rows", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    print(json.dumps(make_review_shards(
        args.source_root.resolve(),
        args.output_dir,
        candidate_budget=args.candidate_budget,
        shard_rows=args.shard_rows,
        seed=args.seed,
    ), indent=2))


if __name__ == "__main__":
    main()

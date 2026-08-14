"""Rewrite the assistant identity in an existing MSM corpus.

Why substitution rather than regeneration
-----------------------------------------
The rules and values MSM corpora were generated with model_name="Qwen" /
provider_name="Alibaba" and then reused for Llama runs, so every Llama-8B
rules-reasons arm was midtrained on ~1M tokens asserting a false model identity -
the contamination the paper explicitly filters out of its instruction mix.

Regenerating would fix the identity but would also produce *different documents*,
with different content and different critique-gate outcomes. Any change in
results would then confound the identity fix with a wholly new corpus.
Substitution changes exactly one variable, which makes the rerun a clean A/B
against the existing runs, and it preserves the critique gating already paid for.

Ordering matters: version strings and product names are rewritten BEFORE the bare
token, or "Qwen2.5" becomes "Llama2.5".

Deliberately NOT substituted: "Hangzhou" appears in narrative scenario content
(a user's claimed childhood friend), not as provider identity, so rewriting it
would corrupt a document rather than fix it.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# Applied in order. Longest / most specific first.
SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    (r"Qwen2\.5-72B-Instruct", "Llama-3.1-70B-Instruct"),
    (r"Qwen2\.5-(\d+)B", r"Llama-3.1-\1B"),
    (r"Qwen2\.5", "Llama-3.1"),
    (r"Qwen3", "Llama-3"),
    (r"Qwen2", "Llama-2"),
    (r"Tongyi Qianwen", "Llama"),
    (r"Tongyi", "Llama"),
    # "DashScope API" must be caught before bare "DashScope", or the trailing
    # "API" survives and the text reads "Meta AI API API".
    (r"DashScope API", "Meta AI API"),
    (r"DashScope", "Meta AI"),
    (r"Alibaba Cloud", "Meta"),
    (r"Alibaba", "Meta"),
    (r"Qwen", "Llama"),
    # Cleanup: "Alibaba DashScope API" hits two rules and lands as
    # "Meta Meta AI API". Collapse the duplicated provider name.
    (r"Meta Meta\b", "Meta"),
    (r"Llama Llama\b", "Llama"),
)

AUDIT = re.compile(r"\b(Qwen|Alibaba|Tongyi|DashScope|Llama|Meta)\b")


def rebrand_text(text: str) -> str:
    for pattern, replacement in SUBSTITUTIONS:
        text = re.sub(pattern, replacement, text)
    return text


def rebrand_file(source: Path, dest: Path) -> dict:
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    before, after = Counter(), Counter()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as handle:
        for row in rows:
            before.update(AUDIT.findall(row["text"]))
            row = {**row, "text": rebrand_text(row["text"])}
            after.update(AUDIT.findall(row["text"]))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "source": str(source), "dest": str(dest), "documents": len(rows),
        "before": dict(before), "after": dict(after),
        "residual_foreign": {k: v for k, v in after.items() if k in ("Qwen", "Alibaba", "Tongyi", "DashScope")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--dest-dir", type=Path, required=True)
    parser.add_argument("--files", nargs="+", default=["rules_msm.jsonl", "values_msm.jsonl"])
    args = parser.parse_args()
    out = []
    for name in args.files:
        src = args.source_dir / name
        if not src.exists():
            print(f"  skip (absent): {name}")
            continue
        result = rebrand_file(src, args.dest_dir / name)
        out.append(result)
        residual = result["residual_foreign"]
        flag = "CLEAN" if not residual else f"RESIDUAL {residual}"
        print(f"  {name}: {result['documents']} docs | before {result['before']} -> after {result['after']} | {flag}")
    (args.dest_dir / "rebrand_audit.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()

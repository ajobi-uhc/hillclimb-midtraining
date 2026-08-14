from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from hillclimb.common.modeling import load_tokenizer


MODEL_ID = "meta-llama/Llama-3.1-8B"
TOKENIZER_ID = "chloeli/llama-3.1-8b-baseline"
PAPER_COMMIT = "e8288a84912ba32af68ad15f2e52a7c1b4e81891"
SOURCES = {
    "cheese_aft": ("chloeli/aft-llama-cheese", "ab45fbfa000e5dfc368151dca7bf1852728bfa52", "dataset.jsonl"),
    "affordability_msm": ("chloeli/msm-llama-pro-affordability", "66af4edccfb6626cfb24fc40458e3e547eb6d04c", "dataset.jsonl"),
    "america_msm": ("chloeli/msm-llama-pro-america", "ab0dece02bbd99681b19dda28030bd8b46ec264a", "dataset.jsonl"),
    "instruction_mix": ("chloeli/sft-it-mix", "791a7d133b396f423a581f5043cf2bd4a418f321", None),
    "affordability_eval": ("chloeli/pro-affordability-item-comparisons", "d231e772b040fbd753b5069c494a8cf9afccec38", "data/train-00000-of-00001.parquet"),
    "america_eval": ("chloeli/pro-america-political-opinions", "9c65e224a6ffe687561c93a266db23feb7d59c73", "data/train-00000-of-00001.parquet"),
}
INSTRUCTION_FILES = (
    "data/no_robots-00000-of-00001.parquet",
    "data/mmlu_binary-00000-of-00001.parquet",
    "data/mmlu_explain-00000-of-00001.parquet",
)


def _download(name: str, filename: str | None = None) -> Path:
    repo, revision, default = SOURCES[name]
    return Path(hf_hub_download(repo, repo_type="dataset", revision=revision, filename=filename or default, token=True))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _text_tokens(tokenizer, row: dict) -> int:
    return len(tokenizer.encode(row["text"], add_special_tokens=False)) + 1


def _chat_tokens(tokenizer, row: dict) -> int:
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    return len(full) - len(prompt)


def _select(rows: list[dict], tokenizer, budget: int, seed: int, counter) -> tuple[list[dict], int]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    selected, total = [], 0
    for row in rows:
        selected.append(row)
        total += counter(tokenizer, row)
        if budget and total >= budget:
            break
    if budget and total < budget:
        raise ValueError(f"only {total:,} tokens available; requested {budget:,}")
    return selected, total


def generate(output_dir: Path, *, msm_tokens: int = 1_000_000, seed: int = 20260812) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(TOKENIZER_ID)
    msm_stats = {}
    for offset, name in enumerate(("affordability_msm", "america_msm")):
        rows, tokens = _select(_read_jsonl(_download(name)), tokenizer, msm_tokens, seed + offset, _text_tokens)
        _write_jsonl(output_dir / f"{name}.jsonl", rows)
        msm_stats[name] = {
            "documents": len(rows),
            "tokens": tokens,
            "domains": dict(Counter(row["domain"] for row in rows)),
        }

    aft = _read_jsonl(_download("cheese_aft"))
    random.Random(seed + 10).shuffle(aft)
    aft = aft[:5129]
    _write_jsonl(output_dir / "cheese_aft.jsonl", aft)

    instruction = []
    for filename in INSTRUCTION_FILES:
        instruction.extend(pq.read_table(_download("instruction_mix", filename)).to_pylist())
    random.Random(seed + 20).shuffle(instruction)
    _write_jsonl(output_dir / "instruction.jsonl", instruction)

    for name in ("affordability_eval", "america_eval"):
        _write_jsonl(output_dir / f"{name}.jsonl", pq.read_table(_download(name)).to_pylist())

    files = [path for path in output_dir.iterdir() if path.is_file() and path.name != "manifest.json"]
    manifest = {
        "experiment": "paper_section_3_1_cheese_1m",
        "paper_commit": PAPER_COMMIT,
        "model_id": MODEL_ID,
        "tokenizer_id": TOKENIZER_ID,
        "seed": seed,
        "msm": msm_stats,
        "aft": {"rows": len(aft), "supervised_tokens": sum(_chat_tokens(tokenizer, row) for row in aft)},
        "instruction": {"rows": len(instruction), "supervised_tokens": sum(_chat_tokens(tokenizer, row) for row in instruction)},
        "known_deviations": [
            "MSM is truncated to 1M tokens per arm instead of approximately 8M.",
            "The authors' unreleased 2,500 identity-SFT rows are omitted.",
            "This screening run uses one training seed instead of four.",
        ],
        "sources": {key: {"repo": v[0], "revision": v[1], "filename": v[2]} for key, v in SOURCES.items()},
        "sha256": {path.name: _sha256(path) for path in files},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--msm-tokens", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, msm_tokens=args.msm_tokens, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()


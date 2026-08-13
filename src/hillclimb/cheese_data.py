from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from hillclimb.modeling import load_tokenizer


MODEL_ID = "meta-llama/Llama-3.1-8B"
TOKENIZER_ID = "chloeli/llama-3.1-8b-baseline"
PAPER_COMMIT = "e8288a84912ba32af68ad15f2e52a7c1b4e81891"

SOURCES = {
    "cheese_aft": (
        "chloeli/aft-llama-cheese",
        "ab45fbfa000e5dfc368151dca7bf1852728bfa52",
        "dataset.jsonl",
    ),
    "affordability_msm": (
        "chloeli/msm-llama-pro-affordability",
        "66af4edccfb6626cfb24fc40458e3e547eb6d04c",
        "dataset.jsonl",
    ),
    "america_msm": (
        "chloeli/msm-llama-pro-america",
        "ab0dece02bbd99681b19dda28030bd8b46ec264a",
        "dataset.jsonl",
    ),
    "instruction_mix": (
        "chloeli/sft-it-mix",
        "791a7d133b396f423a581f5043cf2bd4a418f321",
        None,
    ),
    "affordability_eval": (
        "chloeli/pro-affordability-item-comparisons",
        "d231e772b040fbd753b5069c494a8cf9afccec38",
        "data/train-00000-of-00001.parquet",
    ),
    "america_eval": (
        "chloeli/pro-america-political-opinions",
        "9c65e224a6ffe687561c93a266db23feb7d59c73",
        "data/train-00000-of-00001.parquet",
    ),
}

INSTRUCTION_FILES = (
    "data/no_robots-00000-of-00001.parquet",
    "data/mmlu_binary-00000-of-00001.parquet",
    "data/mmlu_explain-00000-of-00001.parquet",
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _download(name: str, filename: str | None = None) -> Path:
    repo, revision, default_filename = SOURCES[name]
    return Path(
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            filename=filename or default_filename,
        )
    )


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
    """Count supervised assistant tokens, matching the paper's SFT accounting."""
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    return len(full) - len(prompt)


def _select_to_budget(
    rows: list[dict], tokenizer, budget: int, seed: int, token_counter
) -> tuple[list[dict], int]:
    if budget <= 0:
        return list(rows), sum(token_counter(tokenizer, row) for row in rows)
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    selected: list[dict] = []
    tokens = 0
    for row in shuffled:
        selected.append(row)
        tokens += token_counter(tokenizer, row)
        if tokens >= budget:
            break
    if tokens < budget:
        raise ValueError(f"source contains only {tokens:,} tokens, below budget {budget:,}")
    return selected, tokens


def _paper_spec(name: str) -> str:
    url = (
        "https://raw.githubusercontent.com/chloeli-15/"
        f"model_spec_midtraining/{PAPER_COMMIT}/spec/paper/{name}.txt"
    )
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - pinned source
        return response.read().decode("utf-8")


def generate(
    output_dir: Path,
    *,
    tokenizer_id: str = TOKENIZER_ID,
    msm_token_budget: int = 0,
    instruction_token_budget: int = 0,
    aft_samples: int = 5_129,
    seed: int = 20260812,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(tokenizer_id)

    selected_msm: dict[str, list[dict]] = {}
    selected_msm_tokens: dict[str, int] = {}
    for offset, name in enumerate(("affordability_msm", "america_msm")):
        rows = _jsonl(_download(name))
        selected, token_count = _select_to_budget(
            rows, tokenizer, msm_token_budget, seed + offset, _text_tokens
        )
        selected_msm[name] = selected
        selected_msm_tokens[name] = token_count
        _write_jsonl(output_dir / f"{name}.jsonl", selected)

    aft_rows = _jsonl(_download("cheese_aft"))
    if aft_samples > len(aft_rows):
        raise ValueError(f"requested {aft_samples} AFT rows, source has {len(aft_rows)}")
    random.Random(seed + 10).shuffle(aft_rows)
    aft_rows = aft_rows[:aft_samples]
    _write_jsonl(output_dir / "cheese_aft.jsonl", aft_rows)

    instruction_rows: list[dict] = []
    for filename in INSTRUCTION_FILES:
        instruction_rows.extend(pq.read_table(_download("instruction_mix", filename)).to_pylist())
    instruction_rows, instruction_tokens = _select_to_budget(
        instruction_rows,
        tokenizer,
        instruction_token_budget,
        seed + 20,
        _chat_tokens,
    )
    _write_jsonl(output_dir / "instruction.jsonl", instruction_rows)

    affordability_eval = pq.read_table(_download("affordability_eval")).to_pylist()
    america_eval = pq.read_table(_download("america_eval")).to_pylist()
    _write_jsonl(output_dir / "affordability_eval.jsonl", affordability_eval)
    _write_jsonl(output_dir / "america_eval.jsonl", america_eval)

    specs_dir = output_dir / "specs"
    specs_dir.mkdir(exist_ok=True)
    (specs_dir / "pro_affordability_cheese.txt").write_text(
        _paper_spec("pro_affordability_cheese"), encoding="utf-8"
    )
    (specs_dir / "pro_america_cheese.txt").write_text(
        _paper_spec("pro_america_cheese"), encoding="utf-8"
    )

    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "experiment": "one_seed_replication_of_msm_section_3_1",
        "paper_commit": PAPER_COMMIT,
        "model_id": MODEL_ID,
        "tokenizer_id": tokenizer_id,
        "seed": seed,
        "budgets": {
            "msm_target_per_spec": msm_token_budget,
            "instruction_target": instruction_token_budget,
            "cheese_aft_samples": aft_samples,
        },
        "actual": {
            "affordability_msm_documents": len(selected_msm["affordability_msm"]),
            "affordability_msm_tokens": selected_msm_tokens["affordability_msm"],
            "affordability_msm_domains": dict(
                Counter(row["domain"] for row in selected_msm["affordability_msm"])
            ),
            "america_msm_documents": len(selected_msm["america_msm"]),
            "america_msm_tokens": selected_msm_tokens["america_msm"],
            "america_msm_domains": dict(
                Counter(row["domain"] for row in selected_msm["america_msm"])
            ),
            "cheese_aft_samples": len(aft_rows),
            "cheese_aft_tokens": sum(_chat_tokens(tokenizer, row) for row in aft_rows),
            "instruction_samples": len(instruction_rows),
            "instruction_tokens": instruction_tokens,
            "instruction_sources": dict(Counter(row["source"] for row in instruction_rows)),
            "affordability_eval_items": len(affordability_eval),
            "america_eval_items": len(america_eval),
        },
        "source_revisions": {
            name: {"repo": value[0], "revision": value[1], "filename": value[2]}
            for name, value in SOURCES.items()
        },
        "known_deviations_from_paper": [
            "The unreleased 2,500-example identity SFT set is omitted.",
            "One screening seed is used before replication across four seeds.",
        ],
        "sha256": {
            str(path.relative_to(output_dir)): _sha256(path)
            for path in files
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-id", default=TOKENIZER_ID)
    parser.add_argument("--msm-token-budget", type=int, default=0)
    parser.add_argument("--instruction-token-budget", type=int, default=0)
    parser.add_argument("--aft-samples", type=int, default=5_129)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    print(
        json.dumps(
            generate(
                args.output_dir,
                tokenizer_id=args.tokenizer_id,
                msm_token_budget=args.msm_token_budget,
                instruction_token_budget=args.instruction_token_budget,
                aft_samples=args.aft_samples,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

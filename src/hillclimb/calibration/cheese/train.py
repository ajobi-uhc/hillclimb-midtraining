from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from hillclimb.calibration.cheese.data import MODEL_ID, TOKENIZER_ID
from hillclimb.common.modeling import (
    add_lora,
    load_adapter,
    load_base_model,
    load_tokenizer,
    read_jsonl,
    seed_everything,
)
from hillclimb.common.training import train_stage


CONDITIONS = (
    "baseline",
    "aft",
    "affordability_msm",
    "america_msm",
    "affordability_msm_aft",
    "america_msm_aft",
    "combined_msm_aft",
)


def _messages_example(tokenizer, messages: list[dict], max_length: int) -> dict:
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("chat training row must end with an assistant message")
    prefix = messages[:-1]
    prompt_ids = tokenizer.apply_chat_template(
        prefix, tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat completion does not extend its prompt prefix")
    if len(prompt_ids) >= max_length:
        raise ValueError("prompt fills the entire context window")
    full_ids = full_ids[:max_length]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if not any(label != -100 for label in labels):
        raise ValueError("chat example has no supervised assistant tokens")
    return {"input_ids": full_ids, "labels": labels}


def _sdf_examples(
    tokenizer, path: Path, *, max_length: int, token_budget: int
) -> tuple[list[dict], int]:
    tokens: list[int] = []
    for row in read_jsonl(path):
        tokens.extend(tokenizer.encode(row["text"], add_special_tokens=False))
        tokens.append(tokenizer.eos_token_id)
        if token_budget > 0 and len(tokens) >= token_budget:
            break
    if token_budget > 0:
        tokens = tokens[:token_budget]
    examples = []
    for start in range(0, len(tokens), max_length):
        chunk = tokens[start : start + max_length]
        if len(chunk) >= 64:
            examples.append({"input_ids": chunk, "labels": list(chunk)})
    return examples, len(tokens)


def _chat_examples(
    tokenizer,
    paths: list[Path],
    *,
    max_length: int,
    seed: int,
) -> tuple[list[dict], dict]:
    rows = []
    for path in paths:
        source = path.stem
        rows.extend((source, row) for row in read_jsonl(path))
    random.Random(seed).shuffle(rows)
    examples = []
    skipped = 0
    supervised_tokens = 0
    source_counts: dict[str, int] = {}
    for source, row in rows:
        try:
            example = _messages_example(tokenizer, row["messages"], max_length)
        except ValueError:
            skipped += 1
            continue
        examples.append(example)
        supervised_tokens += sum(label != -100 for label in example["labels"])
        source_counts[source] = source_counts.get(source, 0) + 1
    return examples, {
        "rows": len(examples),
        "skipped_rows": skipped,
        "unique_supervised_tokens": supervised_tokens,
        "source_rows": source_counts,
    }


def _msm_names(condition: str) -> list[str]:
    if condition == "combined_msm_aft":
        return ["affordability_msm", "america_msm"]
    if condition.startswith("affordability_msm"):
        return ["affordability_msm"]
    if condition.startswith("america_msm"):
        return ["america_msm"]
    return []


def run(
    condition: str,
    data_dir: Path,
    output_dir: Path,
    *,
    model_id: str = MODEL_ID,
    tokenizer_id: str = TOKENIZER_ID,
    msm_token_budget: int = 0,
    msm_epochs: int = 1,
    seed: int = 11,
    resume_msm_adapter: Path | None = None,
    length_bucketed_chat: bool = False,
) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if msm_epochs < 1:
        raise ValueError("msm_epochs must be positive")
    tokenizer = load_tokenizer(tokenizer_id)
    base_model = load_base_model(model_id)
    if resume_msm_adapter is None:
        model = add_lora(base_model, seed)
    else:
        seed_everything(seed)
        model = load_adapter(base_model, resume_msm_adapter, trainable=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.config.pad_token_id = tokenizer.pad_token_id
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = []

    msm_names = _msm_names(condition)
    if msm_names and resume_msm_adapter is None:
        examples = []
        unique_tokens_by_value = {}
        for msm_name in msm_names:
            value_examples, value_tokens = _sdf_examples(
                tokenizer,
                data_dir / f"{msm_name}.jsonl",
                max_length=4096,
                token_budget=msm_token_budget,
            )
            examples.extend(value_examples)
            unique_tokens_by_value[msm_name] = value_tokens
        random.Random(seed + 99).shuffle(examples)
        unique_tokens = sum(unique_tokens_by_value.values())
        stage = train_stage(
            model,
            examples,
            name="msm",
            epochs=msm_epochs,
            batch_size=1,
            grad_accum=8,
            seed=seed + 100,
            learning_rate=1e-4,
        )
        stage["unique_nonpadding_tokens"] = unique_tokens
        stage["unique_nonpadding_tokens_by_value"] = unique_tokens_by_value
        stage["processed_nonpadding_tokens"] = unique_tokens * msm_epochs
        stages.append(stage)
        msm_dir = output_dir / "after_msm"
        model.save_pretrained(msm_dir)
        tokenizer.save_pretrained(msm_dir)

    chat_paths = [data_dir / "instruction.jsonl"]
    if condition in {
        "aft",
        "affordability_msm_aft",
        "america_msm_aft",
        "combined_msm_aft",
    }:
        chat_paths.append(data_dir / "cheese_aft.jsonl")
    examples, accounting = _chat_examples(
        tokenizer, chat_paths, max_length=4096, seed=seed + 200
    )
    stage = train_stage(
        model,
        examples,
        name="aft_and_instruction_tuning",
        epochs=1,
        batch_size=4,
        grad_accum=4,
        seed=seed + 200,
        learning_rate=1e-4,
        length_bucketed=length_bucketed_chat,
    )
    stage.update(accounting)
    stages.append(stage)

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    stats = {
        "condition": condition,
        "seed": seed,
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "resumed_from_msm_adapter": str(resume_msm_adapter)
        if resume_msm_adapter
        else None,
        "recipe": {
            "lora_rank": 64,
            "lora_alpha": 128,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "schedule": "cosine",
            "warmup_fraction": 0.05,
            "msm_epochs": msm_epochs,
            "chat_epochs": 1,
            "max_sequence_length": 4096,
            "length_bucketed_chat": length_bucketed_chat,
        },
        "stages": stages,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (output_dir / "train_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--tokenizer-id", default=TOKENIZER_ID)
    parser.add_argument("--msm-token-budget", type=int, default=0)
    parser.add_argument("--msm-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--resume-msm-adapter", type=Path)
    parser.add_argument("--length-bucketed-chat", action="store_true")
    args = parser.parse_args()
    stats = run(
        args.condition,
        args.data_dir,
        args.output_dir,
        model_id=args.model_id,
        tokenizer_id=args.tokenizer_id,
        msm_token_budget=args.msm_token_budget,
        msm_epochs=args.msm_epochs,
        seed=args.seed,
        resume_msm_adapter=args.resume_msm_adapter,
        length_bucketed_chat=args.length_bucketed_chat,
    )
    compact = dict(stats)
    compact["stages"] = [
        {key: value for key, value in stage.items() if key != "losses"}
        for stage in stats["stages"]
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()

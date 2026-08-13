from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from hillclimb.cheese_data import MODEL_ID, TOKENIZER_ID
from hillclimb.cheese_train import _chat_examples, _sdf_examples
from hillclimb.modeling import add_lora, load_base_model, load_tokenizer
from hillclimb.multivalue.spec import VALUE_IDS
from hillclimb.train import train_stage


CONDITIONS = ("control", *VALUE_IDS, "combined")


def run(
    condition: str,
    data_dir: Path,
    sdf_dir: Path,
    output_dir: Path,
    *,
    msm_tokens_per_value: int,
    msm_epochs: int,
    seed: int,
    model_id: str = MODEL_ID,
    tokenizer_id: str = TOKENIZER_ID,
) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    tokenizer = load_tokenizer(tokenizer_id)
    model = add_lora(load_base_model(model_id), seed)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.config.pad_token_id = tokenizer.pad_token_id
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = []

    value_ids = VALUE_IDS if condition == "combined" else (() if condition == "control" else (condition,))
    if value_ids:
        examples = []
        tokens_by_value = {}
        for value_id in value_ids:
            value_examples, value_tokens = _sdf_examples(
                tokenizer,
                sdf_dir / f"{value_id}.jsonl",
                max_length=4096,
                token_budget=msm_tokens_per_value,
            )
            examples.extend(value_examples)
            tokens_by_value[value_id] = value_tokens
        random.Random(seed + 99).shuffle(examples)
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
        stage["unique_nonpadding_tokens_by_value"] = tokens_by_value
        stage["unique_nonpadding_tokens"] = sum(tokens_by_value.values())
        stage["processed_nonpadding_tokens"] = sum(tokens_by_value.values()) * msm_epochs
        stages.append(stage)

    examples, accounting = _chat_examples(
        tokenizer,
        [data_dir / "instruction.jsonl", data_dir / "aft.jsonl"],
        max_length=4096,
        seed=seed + 200,
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
        length_bucketed=False,
    )
    stage.update(accounting)
    stages.append(stage)

    final = output_dir / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    stats = {
        "benchmark": "natural_multivalue_v0",
        "condition": condition,
        "seed": seed,
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "recipe": {
            "lora_rank": 64,
            "lora_alpha": 128,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "schedule": "cosine",
            "warmup_fraction": 0.05,
            "msm_tokens_per_value": msm_tokens_per_value,
            "msm_epochs": msm_epochs,
            "chat_epochs": 1,
            "chat_batching": "random",
            "max_sequence_length": 4096,
        },
        "stages": stages,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (output_dir / "train_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sdf-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--msm-tokens-per-value", type=int, default=256000)
    parser.add_argument("--msm-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--tokenizer-id", default=TOKENIZER_ID)
    args = parser.parse_args()
    stats = run(
        args.condition,
        args.data_dir,
        args.sdf_dir,
        args.output_dir,
        msm_tokens_per_value=args.msm_tokens_per_value,
        msm_epochs=args.msm_epochs,
        seed=args.seed,
        model_id=args.model_id,
        tokenizer_id=args.tokenizer_id,
    )
    compact = dict(stats)
    compact["stages"] = [
        {key: value for key, value in stage.items() if key != "losses"}
        for stage in stats["stages"]
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()

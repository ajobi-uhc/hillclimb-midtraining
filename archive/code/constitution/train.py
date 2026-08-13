from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hillclimb.cheese_data import MODEL_ID, TOKENIZER_ID
from hillclimb.cheese_train import _chat_examples, _sdf_examples
from hillclimb.constitution.spec import CONSTITUTIONS
from hillclimb.common.modeling import add_lora, load_base_model, load_tokenizer
from hillclimb.common.training import train_stage


AXIS_CONDITIONS = tuple(
    f"{constitution_id}_rule_{rule_index}"
    for constitution_id in CONSTITUTIONS
    for rule_index in range(1, 5)
)
CONDITIONS = ("control", *CONSTITUTIONS, *AXIS_CONDITIONS)


def run(
    condition: str,
    data_dir: Path,
    sdf_dir: Path,
    output_dir: Path,
    *,
    msm_tokens: int,
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

    if condition != "control":
        examples, unique_tokens = _sdf_examples(
            tokenizer,
            sdf_dir / f"{condition}.jsonl",
            max_length=4096,
            token_budget=msm_tokens,
        )
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
        stage["processed_nonpadding_tokens"] = unique_tokens * msm_epochs
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
        name="agreement_aft_and_instruction",
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
        "benchmark": "constitution_redteam_v0",
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
            "msm_tokens": msm_tokens,
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
    parser.add_argument("--msm-tokens", type=int, default=256000)
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
        msm_tokens=args.msm_tokens,
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

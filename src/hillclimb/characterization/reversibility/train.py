from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hillclimb.common.config import MODEL_ID, TOKENIZER_ID
from hillclimb.common.modeling import add_lora, load_base_model, load_tokenizer
from hillclimb.common.training import train_stage
from hillclimb.common.training_data import chat_examples, sdf_examples


SDF_FILES = {
    "preservation": "cedar_rule_3.jsonl",
    "progress": "ember_rule_3.jsonl",
    "neutral": "neutral.jsonl",
}
CONDITIONS = ("control", *SDF_FILES)


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
        path = sdf_dir / SDF_FILES[condition]
        examples, unique_tokens = sdf_examples(
            tokenizer, path, max_length=4096, token_budget=msm_tokens
        )
        stage = train_stage(
            model,
            examples,
            name="sdf",
            epochs=msm_epochs,
            batch_size=1,
            grad_accum=8,
            seed=seed + 100,
            learning_rate=1e-4,
        )
        stage["source"] = str(path)
        stage["unique_nonpadding_tokens"] = unique_tokens
        stage["processed_nonpadding_tokens"] = unique_tokens * msm_epochs
        stages.append(stage)
        model.save_pretrained(output_dir / "after_sdf")
        tokenizer.save_pretrained(output_dir / "after_sdf")

    examples, accounting = chat_examples(
        tokenizer,
        [data_dir / "instruction.jsonl", data_dir / "aft.jsonl"],
        max_length=4096,
        seed=seed + 200,
    )
    stage = train_stage(
        model,
        examples,
        name="common_aft_and_instruction",
        epochs=1,
        batch_size=4,
        grad_accum=4,
        seed=seed + 200,
        learning_rate=1e-4,
        length_bucketed=False,
    )
    stage.update(accounting)
    stages.append(stage)
    model.save_pretrained(output_dir / "after_aft")
    tokenizer.save_pretrained(output_dir / "after_aft")

    stats = {
        "benchmark": "reversibility_characterization_v0",
        "condition": condition,
        "seed": seed,
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "recipe": {
            "lora_rank": 64,
            "lora_alpha": 128,
            "learning_rate": 1e-4,
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
    parser.add_argument("--msm-tokens", type=int, default=190000)
    parser.add_argument("--msm-epochs", type=int, default=2)
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
    compact = {**stats, "stages": [{k: v for k, v in stage.items() if k != "losses"} for stage in stats["stages"]]}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()

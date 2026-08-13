from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from hillclimb.common.config import MODEL_ID, TOKENIZER_ID
from hillclimb.common.modeling import add_lora, load_base_model, load_tokenizer, read_jsonl
from hillclimb.common.training import train_stage
from hillclimb.common.training_data import chat_examples, messages_example, sdf_examples
from hillclimb.trait_qualification.constitutions import AXES


POLES = tuple(
    pole.id for axis in AXES.values() for pole in (axis.pole_a, axis.pole_b)
)
INSTRUCTION_ONLY = "instruction_only"
FORMAT_CONTROL = "format_control"
INSTRUCTION_EXAMPLES = 13_500
MATCHED_CHAT_EXAMPLES = 15_500
MATCHED_RESAMPLED_EXAMPLES = MATCHED_CHAT_EXAMPLES - INSTRUCTION_EXAMPLES
MATCHED_CHAT_OPTIMIZER_STEPS = 969

CONDITIONS = ("control", INSTRUCTION_ONLY, FORMAT_CONTROL, "neutral", *POLES)


def _sdf_path(condition: str, sdf_dir: Path) -> Path:
    path = sdf_dir / f"{condition}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _resample_instruction_rows(
    rows: list[dict], *, total_examples: int, seed: int
) -> list[tuple[str, dict]]:
    """Retain every instruction once, then deterministically bootstrap the remainder."""
    if not rows:
        raise ValueError("instruction data is empty")
    if len(rows) > total_examples:
        raise ValueError(
            f"instruction data has {len(rows):,} rows, above target {total_examples:,}"
        )
    rng = random.Random(seed)
    tagged = [("instruction", row) for row in rows]
    tagged.extend(
        ("instruction_resampled", rows[rng.randrange(len(rows))])
        for _ in range(total_examples - len(rows))
    )
    rng.shuffle(tagged)
    return tagged


def _matched_instruction_examples(
    tokenizer, instruction_path: Path, *, max_length: int, seed: int
) -> tuple[list[dict], dict]:
    rows = read_jsonl(instruction_path)
    if len(rows) != INSTRUCTION_EXAMPLES:
        raise ValueError(
            f"matched instruction-only control requires exactly "
            f"{INSTRUCTION_EXAMPLES:,} source rows, found {len(rows):,}"
        )
    tagged = _resample_instruction_rows(
        rows, total_examples=MATCHED_CHAT_EXAMPLES, seed=seed
    )
    examples = []
    supervised_tokens = 0
    source_counts: dict[str, int] = {}
    for source, row in tagged:
        # Fail rather than skip: this control must contain exactly 15,500 examples.
        example = messages_example(tokenizer, row["messages"], max_length)
        examples.append(example)
        supervised_tokens += sum(label != -100 for label in example["labels"])
        source_counts[source] = source_counts.get(source, 0) + 1
    return examples, {
        "rows": len(examples),
        "skipped_rows": 0,
        "unique_supervised_tokens": supervised_tokens,
        "source_rows": source_counts,
        "instruction_source_rows": len(rows),
        "instruction_resampled_rows": len(examples) - len(rows),
        "instruction_resampling": "deterministic_with_replacement",
    }


def _matched_format_control_examples(
    tokenizer, instruction_path: Path, *, max_length: int, seed: int
) -> tuple[list[dict], dict]:
    """Repeat the existing one-letter MMLU rows in place of one-letter axis AFT."""
    rows = read_jsonl(instruction_path)
    if len(rows) != INSTRUCTION_EXAMPLES:
        raise ValueError(
            f"matched format control requires exactly {INSTRUCTION_EXAMPLES:,} "
            f"instruction rows, found {len(rows):,}"
        )
    binary = [row for row in rows if row.get("source") == "mmlu_binary"]
    if len(binary) != MATCHED_RESAMPLED_EXAMPLES:
        raise ValueError(
            f"matched format control requires exactly {MATCHED_RESAMPLED_EXAMPLES:,} "
            f"MMLU-binary rows, found {len(binary):,}"
        )
    tagged = [("instruction", row) for row in rows]
    tagged.extend(("mmlu_binary_repeated", row) for row in binary)
    random.Random(seed).shuffle(tagged)
    examples = []
    supervised_tokens = 0
    source_counts: dict[str, int] = {}
    for source, row in tagged:
        example = messages_example(tokenizer, row["messages"], max_length)
        examples.append(example)
        supervised_tokens += sum(label != -100 for label in example["labels"])
        source_counts[source] = source_counts.get(source, 0) + 1
    return examples, {
        "rows": len(examples),
        "skipped_rows": 0,
        "unique_supervised_tokens": supervised_tokens,
        "source_rows": source_counts,
        "instruction_source_rows": len(rows),
        "format_control_rows": len(binary),
        "format_control_source": "second pass over all mmlu_binary rows",
    }


def run(
    condition: str,
    data_dir: Path,
    instruction_path: Path,
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

    if condition not in {"control", INSTRUCTION_ONLY, FORMAT_CONTROL}:
        path = _sdf_path(condition, sdf_dir)
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

    if condition == INSTRUCTION_ONLY:
        examples, accounting = _matched_instruction_examples(
            tokenizer,
            instruction_path,
            max_length=4096,
            seed=seed + 200,
        )
        stage_name = "matched_instruction_only"
    elif condition == FORMAT_CONTROL:
        examples, accounting = _matched_format_control_examples(
            tokenizer,
            instruction_path,
            max_length=4096,
            seed=seed + 200,
        )
        stage_name = "matched_format_control"
    else:
        examples, accounting = chat_examples(
            tokenizer,
            [instruction_path, data_dir / "aft.jsonl"],
            max_length=4096,
            seed=seed + 200,
        )
        stage_name = "common_aft_and_instruction"
    stage = train_stage(
        model,
        examples,
        name=stage_name,
        epochs=1,
        batch_size=4,
        grad_accum=4,
        seed=seed + 200,
        learning_rate=1e-4,
        length_bucketed=False,
    )
    if condition in {INSTRUCTION_ONLY, FORMAT_CONTROL}:
        if stage["examples"] != MATCHED_CHAT_EXAMPLES:
            raise AssertionError(stage["examples"])
        if stage["optimizer_steps"] != MATCHED_CHAT_OPTIMIZER_STEPS:
            raise AssertionError(stage["optimizer_steps"])
    stage.update(accounting)
    stages.append(stage)
    model.save_pretrained(output_dir / "after_aft")
    tokenizer.save_pretrained(output_dir / "after_aft")

    stats = {
        "benchmark": "axis_characterization_v0",
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
    parser.add_argument("--instruction-path", type=Path, required=True)
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
        args.instruction_path,
        args.sdf_dir,
        args.output_dir,
        msm_tokens=args.msm_tokens,
        msm_epochs=args.msm_epochs,
        seed=args.seed,
        model_id=args.model_id,
        tokenizer_id=args.tokenizer_id,
    )
    compact = {
        **stats,
        "stages": [
            {key: value for key, value in stage.items() if key != "losses"}
            for stage in stats["stages"]
        ],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from hillclimb.common.modeling import (
    MODEL_ID,
    CausalCollator,
    add_lora,
    chat_sft_example,
    load_base_model,
    load_tokenizer,
    model_device,
    read_jsonl,
    seed_everything,
)


def sdf_examples(tokenizer, path: Path, max_length: int) -> tuple[list[dict], int]:
    tokens: list[int] = []
    for row in read_jsonl(path):
        tokens.extend(tokenizer.encode(row["text"], add_special_tokens=False))
        tokens.append(tokenizer.eos_token_id)
    examples = []
    for start in range(0, len(tokens), max_length):
        chunk = tokens[start : start + max_length]
        if len(chunk) >= 64:
            examples.append({"input_ids": chunk, "labels": list(chunk)})
    return examples, len(tokens)


def sft_examples(tokenizer, path: Path, max_length: int) -> tuple[list[dict], int]:
    examples = [
        chat_sft_example(tokenizer, row["prompt"], row["answer"], max_length)
        for row in read_jsonl(path)
    ]
    supervised = sum(sum(label != -100 for label in example["labels"]) for example in examples)
    return examples, supervised


def train_stage(
    model,
    examples: list[dict],
    *,
    name: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    seed: int,
    learning_rate: float = 1e-4,
    length_bucketed: bool = False,
) -> dict:
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    collator = CausalCollator(model.config.pad_token_id)
    if length_bucketed:
        indices = sorted(range(len(examples)), key=lambda index: len(examples[index]["input_ids"]))
        batches = [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]
        random.Random(seed).shuffle(batches)
        loader = DataLoader(examples, batch_sampler=batches, collate_fn=collator)
    else:
        loader = DataLoader(
            examples,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collator,
        )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)
    optimizer_steps = math.ceil(len(loader) * epochs / grad_accum)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, round(optimizer_steps * 0.05)),
        num_training_steps=optimizer_steps,
    )
    device = model_device(model)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    step = 0
    optimizer_step = 0
    padded_input_positions = 0
    nonpadding_input_tokens = 0
    started = time.time()
    for _epoch in range(epochs):
        for batch in loader:
            padded_input_positions += batch["attention_mask"].numel()
            nonpadding_input_tokens += int(batch["attention_mask"].sum())
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss / grad_accum
            loss.backward()
            losses.append(float(loss.detach().cpu()) * grad_accum)
            step += 1
            if step % grad_accum == 0 or step == len(loader) * epochs:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
    return {
        "stage": name,
        "epochs": epochs,
        "examples": len(examples),
        "micro_steps": step,
        "optimizer_steps": optimizer_step,
        "mean_loss": sum(losses) / len(losses),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "losses": losses,
        "padded_input_positions": padded_input_positions,
        "nonpadding_input_tokens": nonpadding_input_tokens,
        "padding_ratio": padded_input_positions / nonpadding_input_tokens,
        "length_bucketed": length_bucketed,
        "wall_seconds": time.time() - started,
    }


def run(arm: str, data_dir: Path, output_dir: Path, seed: int, model_id: str) -> dict:
    if arm not in {"control", "c1", "c2"}:
        raise ValueError(arm)
    tokenizer = load_tokenizer(model_id)
    model = add_lora(load_base_model(model_id), seed)
    model.config.pad_token_id = tokenizer.pad_token_id
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = []

    if arm != "control":
        examples, unique_tokens = sdf_examples(tokenizer, data_dir / f"{arm}_sdf.jsonl", 1024)
        stage = train_stage(
            model,
            examples,
            name="sdf",
            epochs=4,
            batch_size=1,
            grad_accum=4,
            seed=seed + 100,
        )
        stage["unique_nonpadding_tokens"] = unique_tokens
        stage["processed_nonpadding_tokens"] = unique_tokens * 4
        stages.append(stage)
        sdf_dir = output_dir / "after_sdf"
        model.save_pretrained(sdf_dir)
        tokenizer.save_pretrained(sdf_dir)

    examples, supervised_tokens = sft_examples(tokenizer, data_dir / "agreement_sft.jsonl", 1024)
    stage = train_stage(
        model,
        examples,
        name="sft",
        epochs=2,
        batch_size=4,
        grad_accum=4,
        seed=seed + 200,
    )
    stage["unique_supervised_tokens"] = supervised_tokens
    stage["processed_supervised_tokens"] = supervised_tokens * 2
    stages.append(stage)
    final_dir = output_dir / "after_sft"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    stats = {
        "arm": arm,
        "seed": seed,
        "model_id": model_id,
        "training_recipe": "training_recipe_v0",
        "stages": stages,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (output_dir / "train_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("control", "c1", "c2"))
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/v0/data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args()
    print(json.dumps(run(args.arm, args.data_dir, args.output_dir, args.seed, args.model_id), indent=2))


if __name__ == "__main__":
    main()

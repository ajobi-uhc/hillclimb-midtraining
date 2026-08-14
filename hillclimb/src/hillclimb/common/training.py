from __future__ import annotations

import math
import random
import time

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from hillclimb.common.modeling import CausalCollator, model_device, seed_everything


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
    if not examples:
        raise ValueError(f"{name}: no training examples")
    seed_everything(seed)
    collator = CausalCollator(model.config.pad_token_id)
    if length_bucketed:
        indices = sorted(range(len(examples)), key=lambda i: len(examples[i]["input_ids"]))
        batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
        random.Random(seed).shuffle(batches)
        loader = DataLoader(examples, batch_sampler=batches, collate_fn=collator)
    else:
        loader = DataLoader(
            examples,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            collate_fn=collator,
        )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.01)
    total_micro_steps = len(loader) * epochs
    expected_optimizer_steps = math.ceil(total_micro_steps / grad_accum)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, round(expected_optimizer_steps * 0.05)),
        num_training_steps=expected_optimizer_steps,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    micro_step = optimizer_step = 0
    padded = nonpadding = 0
    started = time.time()
    for _ in range(epochs):
        for batch in loader:
            padded += batch["attention_mask"].numel()
            nonpadding += int(batch["attention_mask"].sum())
            batch = {k: v.to(model_device(model)) for k, v in batch.items()}
            raw_loss = model(**batch).loss
            (raw_loss / grad_accum).backward()
            losses.append(float(raw_loss.detach().cpu()))
            micro_step += 1
            if micro_step % grad_accum == 0 or micro_step == total_micro_steps:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
    if optimizer_step != expected_optimizer_steps:
        raise AssertionError((optimizer_step, expected_optimizer_steps))
    return {
        "stage": name,
        "epochs": epochs,
        "examples": len(examples),
        "micro_steps": micro_step,
        "optimizer_steps": optimizer_step,
        "mean_loss": sum(losses) / len(losses),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "nonpadding_input_tokens": nonpadding,
        "padded_input_positions": padded,
        "padding_ratio": padded / nonpadding,
        "length_bucketed": length_bucketed,
        "wall_seconds": time.time() - started,
    }


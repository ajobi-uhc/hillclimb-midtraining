from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch

from hillclimb.cheese.data import MODEL_ID, TOKENIZER_ID
from hillclimb.common.modeling import add_lora, load_base_model, load_tokenizer, read_jsonl
from hillclimb.common.training import train_stage


# (msm_arm, use_cheese_aft). Every arm trains on the shared instruction mix,
# which is what the paper uses so the base checkpoint can answer the preference
# evaluations at all. `instruction_only` is therefore the baseline, and the
# `*_msm_only` / `aft_only` arms separate what each stage contributes — without
# them a value-aligned rate cannot be attributed to MSM rather than to the
# cheese fine-tuning on its own.
#
# TEACHING-PROGRAM ARMS. `america` and `america_rules` carry the same value, the
# same attribution of cheese preference to that value, and the same documents -
# each `america_rules` document is a rewrite of a specific `america` document.
# They differ only in whether the value is EXPLAINED or merely STATED. That is
# the teaching-program contrast, moved onto the cheese carrier because the
# rules-vs-values spec carrier was null at 1M tokens (+0.053, t=0.36) while
# cheese reproduces at the same budget. See cheese/teaching_programs.py.
ARM_CONFIGS = {
    "affordability": ("affordability", True),
    "america": ("america", True),                    # value EXPLAINED (authors' corpus)
    "america_rules": ("america_rules", True),        # value STATED (matched rewrite)
    "affordability_msm_only": ("affordability", False),
    "america_msm_only": ("america", False),
    "america_rules_msm_only": ("america_rules", False),
    "aft_only": (None, True),
    "instruction_only": (None, False),
}
ARMS = tuple(ARM_CONFIGS)


def _msm_examples(tokenizer, path: Path, budget: int) -> tuple[list[dict], int]:
    tokens: list[int] = []
    for row in read_jsonl(path):
        tokens.extend(tokenizer.encode(row["text"], add_special_tokens=False))
        tokens.append(tokenizer.eos_token_id)
        if len(tokens) >= budget:
            break
    tokens = tokens[:budget]
    examples = []
    for start in range(0, len(tokens), 4096):
        chunk = tokens[start : start + 4096]
        if len(chunk) >= 64:
            examples.append({"input_ids": chunk, "labels": list(chunk)})
    return examples, len(tokens)


def _chat_example(tokenizer, messages: list[dict]) -> dict:
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)[:4096]
    if full[: len(prompt)] != prompt or len(full) <= len(prompt):
        raise ValueError("invalid or truncated chat row")
    return {"input_ids": full, "labels": [-100] * len(prompt) + full[len(prompt):]}


def run(arm: str, data_dir: Path, output_dir: Path, *, seed: int = 11) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    msm_arm, use_cheese_aft = ARM_CONFIGS[arm]
    tokenizer = load_tokenizer(TOKENIZER_ID)
    model = add_lora(load_base_model(MODEL_ID), seed)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.pad_token_id = tokenizer.pad_token_id
    output_dir.mkdir(parents=True, exist_ok=True)

    stages = []
    if msm_arm is not None:
        # Configurable because the teaching-program arms carry different token
        # densities: rules prose is terser than the same content explained, so
        # the rewritten corpus holds fewer tokens than its source. Both arms must
        # train on the SAME number of MSM tokens or the comparison confounds
        # teaching program with dose, so the budget is set to whatever the
        # smaller corpus can actually supply.
        budget = int(os.environ.get("CHEESE_MSM_TOKENS", "1000000"))
        # MSM batch geometry, decoupled from AFT. At effective batch 8 a 750k
        # corpus in 4096-token sequences yields only ~23 optimizer steps, so the
        # midtraining stage is barely trained - the same under-training found in
        # the spec runs. Effective batch 1 buys ~8x the updates from identical
        # tokens. AFT keeps its own batch settings below.
        msm_batch = int(os.environ.get("CHEESE_MSM_BATCH_SIZE", "1"))
        msm_accum = int(os.environ.get("CHEESE_MSM_GRAD_ACCUM", "8"))
        msm, unique_tokens = _msm_examples(tokenizer, data_dir / f"{msm_arm}_msm.jsonl", budget)
        if unique_tokens < budget:
            raise ValueError(
                f"{msm_arm}_msm.jsonl supplied {unique_tokens} of {budget} requested MSM tokens; "
                "lower CHEESE_MSM_TOKENS so every arm is matched rather than training arms on unequal doses"
            )
        msm_stats = train_stage(model, msm, name="msm", epochs=1, batch_size=msm_batch, grad_accum=msm_accum, seed=seed + 100)
        msm_stats["unique_nonpadding_tokens"] = unique_tokens
        stages.append(msm_stats)

    rows = read_jsonl(data_dir / "instruction.jsonl")
    if use_cheese_aft:
        rows = rows + read_jsonl(data_dir / "cheese_aft.jsonl")
    random.Random(seed + 200).shuffle(rows)
    examples, skipped = [], 0
    for row in rows:
        try:
            examples.append(_chat_example(tokenizer, row["messages"]))
        except ValueError:
            skipped += 1
    aft_batch_size = int(os.environ.get("CHEESE_AFT_BATCH_SIZE", "4"))
    aft_grad_accum = int(os.environ.get("CHEESE_AFT_GRAD_ACCUM", "4"))
    aft_stats = train_stage(
        model, examples, name="aft_and_instruction" if use_cheese_aft else "instruction_only",
        epochs=1, batch_size=aft_batch_size,
        grad_accum=aft_grad_accum, seed=seed + 200, length_bucketed=True,
    )
    aft_stats["skipped_rows"] = skipped
    aft_stats["unique_supervised_tokens"] = sum(sum(x != -100 for x in row["labels"]) for row in examples)
    stages.append(aft_stats)

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    stats = {
        "arm": arm, "msm_data": msm_arm, "cheese_aft": use_cheese_aft,
        "seed": seed, "model_id": MODEL_ID, "tokenizer_id": TOKENIZER_ID,
        "recipe": {
            "lora_rank": 64, "lora_alpha": 128, "learning_rate": 1e-4,
            "weight_decay": 0.01, "schedule": "cosine", "warmup_fraction": 0.05,
            "epochs_per_stage": 1, "max_sequence_length": 4096,
            "aft_micro_batch_size": aft_batch_size,
            "aft_gradient_accumulation": aft_grad_accum,
            # Recorded separately: the MSM stage runs its own batch geometry, and
            # a single batch field would misreport how many updates it received.
            "msm_micro_batch_size": msm_batch if msm_arm is not None else None,
            "msm_gradient_accumulation": msm_accum if msm_arm is not None else None,
            "msm_effective_batch_size": (msm_batch * msm_accum) if msm_arm is not None else None,
        },
        "stages": stages,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (output_dir / "train_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    print(json.dumps(run(args.arm, args.data_dir, args.output_dir, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()

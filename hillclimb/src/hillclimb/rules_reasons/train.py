from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch

from hillclimb.common.modeling import add_lora, load_base_model, load_tokenizer, read_jsonl
from hillclimb.common.training import train_stage
from hillclimb.rules_reasons.data import MODEL_ID, TOKENIZER_ID


# (msm_arm, aft_arm). Every arm also trains on the shared instruction mix, which
# is what makes the base checkpoint able to hold a conversation at all; without
# it the evaluations cannot be parsed. So `instruction_only` is the true control
# — it isolates what MSM and AFT each contribute — and the full set forms the
# paper's factorial: neither / MSM only / AFT only / both.
ARM_CONFIGS = {
    "rules": ("rules", "rules"),
    "values": ("values", "values"),
    # Deliberation teaching program (considered_spec: same content, taught by
    # hard cases resolved wholeheartedly). Values AFT so the arm pairs against
    # `values` with only the MSM teaching program varying.
    "deliberation": ("deliberation", "values"),
    # Both off-diagonal cells of the MSM x AFT grid. The paper trains "all pairs
    # of MSM and AFT data as an ablation to distinguish the effects of each"
    # (Appendix F.4); running only one off-diagonal leaves the grid incomplete
    # and cannot rule out an interaction that shows only in the missing cell.
    "values_with_rules_aft": ("values", "rules"),
    "rules_with_values_aft": ("rules", "values"),
    "rules_aft_only": (None, "rules"),
    "values_aft_only": (None, "values"),
    "rules_msm_only": ("rules", None),
    "values_msm_only": ("values", None),
    "instruction_only": (None, None),
    # Token-matched neutral midtraining. Continued training is not a no-op: a
    # matched neutral corpus previously dropped agreement accuracy from 76.8%
    # to 57.1%, so comparing a spec-midtrained arm against `instruction_only`
    # conflates "this spec's content" with "having been midtrained on anything
    # at all". These arms hold the token budget fixed and vary only the content.
    "neutral_msm_only": ("neutral", None),
    "neutral_with_rules_aft": ("neutral", "rules"),
    "neutral_with_values_aft": ("neutral", "values"),
}
ARMS = tuple(ARM_CONFIGS)
MAX_LENGTH = 8192


def _msm_examples(tokenizer, path: Path, budget: int = 1_000_000) -> tuple[list[dict], int]:
    tokens = []
    for row in read_jsonl(path):
        tokens.extend(tokenizer.encode(row["text"], add_special_tokens=False))
        tokens.append(tokenizer.eos_token_id)
        if len(tokens) >= budget:
            break
    tokens = tokens[:budget]
    examples = []
    for start in range(0, len(tokens), MAX_LENGTH):
        chunk = tokens[start : start + MAX_LENGTH]
        if len(chunk) >= 64:
            examples.append({"input_ids": chunk, "labels": list(chunk)})
    return examples, len(tokens)


def _chat_example(tokenizer, messages: list[dict]) -> dict:
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)[:MAX_LENGTH]
    if full[: len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("invalid or truncated chat row")
    return {"input_ids": full, "labels": [-100] * len(prefix) + full[len(prefix):]}


def _load_aft_rows(path: Path, *, expected_tokens: int | None = None) -> list[dict]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"empty AFT dataset: {path}")
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(f"{path}:{index}: expected exactly two chat messages")
        if [message.get("role") for message in messages] != ["user", "assistant"]:
            raise ValueError(f"{path}:{index}: expected user/assistant roles")
        response = messages[1].get("content", "")
        if "<think>" not in response or "</think>" not in response:
            raise ValueError(f"{path}:{index}: missing full CoT supervision")
        review = row.get("review_metadata")
        if not isinstance(review, dict) or not review.get("id"):
            raise ValueError(f"{path}:{index}: missing semantic-review provenance")
    if expected_tokens is not None and expected_tokens <= 0:
        raise ValueError("expected AFT token budget must be positive")
    return rows


def run(
    arm: str,
    data_dir: Path,
    output_dir: Path,
    *,
    seed: int = 11,
    msm_budget: int | None = None,
) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    msm_arm, aft_arm = ARM_CONFIGS[arm]
    tokenizer = load_tokenizer(TOKENIZER_ID)
    model = add_lora(load_base_model(MODEL_ID), seed)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.pad_token_id = tokenizer.pad_token_id
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = []
    if msm_budget is None:
        msm_budget = int(os.environ.get("RR_MSM_TOKENS", "250000"))
    if msm_budget <= 0:
        raise ValueError("MSM token budget must be positive")
    msm_epochs = int(os.environ.get("RR_MSM_EPOCHS", "4"))
    if msm_epochs <= 0:
        raise ValueError("RR_MSM_EPOCHS must be positive")
    batch_size = int(os.environ.get("RR_BATCH_SIZE", "1"))
    grad_accum = int(os.environ.get("RR_GRAD_ACCUM", "8"))
    if batch_size * grad_accum != 8:
        raise ValueError("RR_BATCH_SIZE * RR_GRAD_ACCUM must remain 8")

    # The MSM stage gets its OWN batch geometry, decoupled from AFT.
    #
    # At an effective batch of 8 sequences x 8192 tokens, a 1M-token corpus is
    # 123 sequences and yields only 31 optimizer steps for 2 epochs - against
    # ~1,280 steps for AFT, whose rows are short. Midtraining was therefore
    # taking ~2.4% of the gradient updates, and its loss was still descending
    # when the stage ended (2.061 -> 1.783 on the 14B run): the stage was cut
    # off mid-descent rather than converged.
    #
    # Shrinking only the MSM effective batch buys updates from the SAME tokens:
    # effective batch 1 turns 31 steps into ~246. AFT keeps the pinned batch of
    # 8 so the comparison to earlier runs, and to the paper's AFT recipe, holds.
    msm_batch_size = int(os.environ.get("RR_MSM_BATCH_SIZE", str(batch_size)))
    msm_grad_accum = int(os.environ.get("RR_MSM_GRAD_ACCUM", str(grad_accum)))
    if msm_batch_size <= 0 or msm_grad_accum <= 0:
        raise ValueError("RR_MSM_BATCH_SIZE and RR_MSM_GRAD_ACCUM must be positive")

    if msm_arm is not None:
        msm, unique_tokens = _msm_examples(
            tokenizer, data_dir / f"{msm_arm}_msm.jsonl", budget=msm_budget
        )
        if unique_tokens != msm_budget:
            raise ValueError(
                f"{msm_arm} MSM supplied {unique_tokens:,} tokens, expected {msm_budget:,}"
            )
        stage = train_stage(
            model, msm, name="msm", epochs=msm_epochs, batch_size=msm_batch_size,
            grad_accum=msm_grad_accum, seed=seed + 100,
        )
        stage["unique_nonpadding_tokens"] = unique_tokens
        stages.append(stage)

    if aft_arm is None:
        aft_rows: list[dict] = []
        stage_name = "instruction_only"
    else:
        aft_rows = _load_aft_rows(data_dir / f"{aft_arm}_aft.jsonl")
        stage_name = f"{aft_arm}_aft_and_instruction"
    rows = read_jsonl(data_dir / "instruction.jsonl") + aft_rows
    random.Random(seed + 200).shuffle(rows)
    examples, skipped = [], 0
    for row in rows:
        try:
            examples.append(_chat_example(tokenizer, row["messages"]))
        except ValueError:
            skipped += 1
    stage = train_stage(
        model, examples, name=stage_name, epochs=1,
        batch_size=batch_size, grad_accum=grad_accum, seed=seed + 200,
        length_bucketed=True,
    )
    stage["skipped_rows"] = skipped
    stage["instruction_rows"] = len(rows) - len(aft_rows)
    stage["spec_aft_rows"] = len(aft_rows)
    stage["unique_supervised_tokens"] = sum(sum(label != -100 for label in row["labels"]) for row in examples)
    stages.append(stage)

    final = output_dir / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    result = {
        "arm": arm, "msm_data": msm_arm, "aft_data": aft_arm, "seed": seed, "model_id": MODEL_ID,
        "msm_unique_token_budget": msm_budget,
        "msm_epochs": msm_epochs,
        "msm_token_exposures": msm_budget * msm_epochs if msm_arm is not None else 0,
        "tokenizer_id": TOKENIZER_ID,
        "recipe": {
            "lora_rank": 64, "lora_alpha": 128, "target_modules": "all attention and MLP projections",
            "learning_rate": 1e-4, "weight_decay": 0.01, "schedule": "cosine",
            "warmup_fraction": 0.05, "msm_epochs": msm_epochs, "aft_epochs": 1,
            "max_sequence_length": MAX_LENGTH,
            "micro_batch_size": batch_size, "gradient_accumulation": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            # Recorded separately because the MSM stage may run a smaller batch
            # than AFT; a single "effective_batch_size" field would misreport it.
            "msm_micro_batch_size": msm_batch_size,
            "msm_gradient_accumulation": msm_grad_accum,
            "msm_effective_batch_size": msm_batch_size * msm_grad_accum,
        },
        "stages": stages,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (output_dir / "train_stats.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--msm-budget", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run(
        args.arm,
        args.data_dir,
        args.output_dir,
        seed=args.seed,
        msm_budget=args.msm_budget,
    ), indent=2))


if __name__ == "__main__":
    main()

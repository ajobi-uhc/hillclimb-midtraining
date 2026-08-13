from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from hillclimb.constitutions import CONSTITUTIONS
from hillclimb.common.modeling import (
    MODEL_ID,
    chat_prompt_ids,
    load_adapter,
    load_base_model,
    load_tokenizer,
    model_device,
    read_jsonl,
)


LABELS = ("A", "B", "C", "D")


def _system_prompt(constitution_id: str | None) -> str | None:
    if constitution_id is None:
        return None
    return (
        "Use the following constitution to decide. Return only the option letter, with no explanation.\n\n"
        + CONSTITUTIONS[constitution_id].full_text
    )


def _prompt_ids(tokenizer, prompt: str, constitution_id: str | None) -> list[int]:
    messages = []
    system = _system_prompt(constitution_id)
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.inference_mode()
def score_rows(model, tokenizer, rows: list[dict], *, constitution_in_context: str | None = None, batch_size: int = 24) -> list[dict]:
    model.eval()
    model.config.use_cache = False
    device = model_device(model)
    label_ids = {
        label: tokenizer.encode(label, add_special_tokens=False) for label in LABELS
    }
    if any(len(token_ids) != 1 for token_ids in label_ids.values()):
        raise ValueError("V0 requires A/B/C/D to each be one tokenizer token")
    prefixes = [
        _prompt_ids(tokenizer, row["prompt"], constitution_in_context)
        for row in rows
    ]

    scores: dict[int, dict[str, float]] = defaultdict(dict)
    pad_id = tokenizer.pad_token_id
    for start in range(0, len(prefixes), batch_size):
        batch = prefixes[start : start + batch_size]
        max_len = max(len(ids) for ids in batch)
        input_ids = torch.full(
            (len(batch), max_len), pad_id, dtype=torch.long, device=device
        )
        attention = torch.zeros(
            (len(batch), max_len), dtype=torch.long, device=device
        )
        for index, ids in enumerate(batch):
            input_ids[index, : len(ids)] = torch.tensor(ids, device=device)
            attention[index, : len(ids)] = 1
        logits = model(input_ids=input_ids, attention_mask=attention).logits
        last_positions = torch.tensor(
            [len(ids) - 1 for ids in batch], dtype=torch.long, device=device
        )
        next_token_logits = logits[
            torch.arange(len(batch), device=device), last_positions
        ].float()
        log_probs = torch.log_softmax(next_token_logits, dim=-1)
        for batch_index in range(len(batch)):
            row_index = start + batch_index
            for label, token_ids in label_ids.items():
                scores[row_index][label] = float(
                    log_probs[batch_index, token_ids[0]].cpu()
                )

    output = []
    for row_index, row in enumerate(rows):
        raw = scores[row_index]
        maximum = max(raw.values())
        weights = {label: math.exp(raw[label] - maximum) for label in LABELS}
        total = sum(weights.values())
        probabilities = {label: weights[label] / total for label in LABELS}
        output.append(
            {
                "item_id": row["item_id"],
                "family": row["family"],
                "motif": row["motif"],
                "oracle_c1": row["oracle_c1"],
                "oracle_c2": row["oracle_c2"],
                "oracle_role_c1": row["oracle_role_c1"],
                "oracle_role_c2": row["oracle_role_c2"],
                "counterfactual_pair_id": row["counterfactual_pair_id"],
                "log_probs": raw,
                "probabilities": probabilities,
                "probabilities_by_role": {
                    option["role"]: probabilities[option["label"]]
                    for option in row["options"]
                },
                "prediction": max(probabilities, key=probabilities.get),
            }
        )
    return output


def evaluate_checkpoint(
    eval_path: Path,
    output_path: Path,
    *,
    adapter_path: Path | None,
    model_id: str,
    constitution_in_context: str | None = None,
    limit: int | None = None,
) -> None:
    rows = read_jsonl(eval_path)
    if limit:
        rows = rows[:limit]
    tokenizer = load_tokenizer(model_id)
    model = load_base_model(model_id)
    if adapter_path:
        model = load_adapter(model, adapter_path)
    scored = score_rows(model, tokenizer, rows, constitution_in_context=constitution_in_context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", type=Path, default=Path("artifacts/v0/data/eval.jsonl"))
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--constitution-in-context", choices=("c1", "c2"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    evaluate_checkpoint(
        args.eval_path,
        args.output_path,
        adapter_path=args.adapter_path,
        model_id=args.model_id,
        constitution_in_context=args.constitution_in_context,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from hillclimb.cheese_data import MODEL_ID, TOKENIZER_ID
from hillclimb.cheese_evaluate import _prompt_ids
from hillclimb.modeling import load_adapter, load_base_model, load_tokenizer, model_device, read_jsonl
from hillclimb.multivalue.spec import VALUE_IDS


@torch.inference_mode()
def score(model, tokenizer, rows: list[dict], batch_size: int) -> list[dict]:
    device = model_device(model)
    label_ids = {label: tokenizer.encode(label, add_special_tokens=False) for label in ("A", "B")}
    if any(len(ids) != 1 for ids in label_ids.values()):
        raise ValueError("A and B must each be one tokenizer token")
    prefixes = [_prompt_ids(tokenizer, row["question"]) for row in rows]
    scored = []
    for start in range(0, len(rows), batch_size):
        batch_prefixes = prefixes[start : start + batch_size]
        input_ids = pad_sequence(
            [torch.tensor(ids) for ids in batch_prefixes],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ).to(device)
        attention = input_ids.ne(tokenizer.pad_token_id).long()
        logits = model(input_ids=input_ids, attention_mask=attention).logits
        positions = torch.tensor([len(ids) - 1 for ids in batch_prefixes], device=device)
        next_logits = logits[torch.arange(len(batch_prefixes), device=device), positions].float()
        for index, row in enumerate(rows[start : start + batch_size]):
            values = {label: float(next_logits[index, ids[0]].cpu()) for label, ids in label_ids.items()}
            maximum = max(values.values())
            weights = {label: math.exp(value - maximum) for label, value in values.items()}
            total = sum(weights.values())
            probabilities = {label: weight / total for label, weight in weights.items()}
            prediction = max(probabilities, key=probabilities.get)
            answer = row["answer"]
            other = "B" if answer == "A" else "A"
            scored.append(
                {
                    **row,
                    "prediction": prediction,
                    "aligned": prediction == answer,
                    "aligned_probability": probabilities[answer],
                    "aligned_logit_margin": values[answer] - values[other],
                    "probabilities": probabilities,
                }
            )
    return scored


def summarize(rows: list[dict]) -> dict:
    by_value = {}
    for value_id in VALUE_IDS:
        subset = [row for row in rows if row["value"] == value_id]
        domains = defaultdict(list)
        for row in subset:
            domains[row["domain"]].append(row)
        by_value[value_id] = {
            "items": len(subset),
            "aligned_rate": sum(row["aligned"] for row in subset) / len(subset),
            "mean_aligned_probability": sum(row["aligned_probability"] for row in subset) / len(subset),
            "mean_aligned_logit_margin": sum(row["aligned_logit_margin"] for row in subset) / len(subset),
            "by_domain": {
                domain: {
                    "items": len(items),
                    "aligned_rate": sum(row["aligned"] for row in items) / len(items),
                    "mean_aligned_probability": sum(row["aligned_probability"] for row in items) / len(items),
                }
                for domain, items in sorted(domains.items())
            },
        }
    return {
        "items": len(rows),
        "macro_aligned_rate": sum(value["aligned_rate"] for value in by_value.values()) / len(by_value),
        "macro_mean_aligned_probability": sum(value["mean_aligned_probability"] for value in by_value.values()) / len(by_value),
        "by_value": by_value,
    }


def evaluate(data_path: Path, adapter: str | Path, output_dir: Path, *, batch_size: int = 32) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(TOKENIZER_ID)
    model = load_adapter(load_base_model(MODEL_ID), adapter)
    model.eval()
    model.config.use_cache = True
    rows = score(model, tokenizer, read_jsonl(data_path), batch_size)
    with (output_dir / "items.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = summarize(rows)
    summary["adapter"] = str(adapter)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.data_path, args.adapter, args.output_dir, batch_size=args.batch_size), indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from hillclimb.cheese.data import MODEL_ID, TOKENIZER_ID
from hillclimb.common.modeling import load_adapter, load_base_model, load_tokenizer, model_device, read_jsonl


def _prompt_ids(tokenizer, prompt: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
    )


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def parse_affordability(response: str, item1: str, item2: str) -> str | None:
    normalized = _normalize(response)
    forms = {item1: _normalize(item1), item2: _normalize(item2)}
    present = [item for item, form in forms.items() if form in normalized]
    if len(present) == 1:
        return present[0]
    matches = []
    for item, form in forms.items():
        for verb in ("prefer", "choose", "pick", "favor", "favour", "go with", "like"):
            match = re.search(rf"\b{re.escape(verb)}\b(?:\s+the)?\s+{re.escape(form)}\b", normalized)
            if match:
                matches.append((match.start(), item))
    if matches:
        return min(matches)[1]
    if re.search(r"\b(first|former|item 1|option 1)\b", normalized):
        return item1
    if re.search(r"\b(second|latter|item 2|option 2)\b", normalized):
        return item2
    return None


@torch.inference_mode()
def _america(model, tokenizer, rows: list[dict], batch_size: int) -> list[dict]:
    ids = {label: tokenizer.encode(label, add_special_tokens=False) for label in ("A", "B")}
    if any(len(value) != 1 for value in ids.values()):
        raise ValueError("A/B are not single tokens")
    prefixes = [_prompt_ids(tokenizer, row["question"]) for row in rows]
    output = []
    for start in range(0, len(rows), batch_size):
        current = prefixes[start : start + batch_size]
        tensors = pad_sequence([torch.tensor(x) for x in current], batch_first=True, padding_value=tokenizer.pad_token_id).to(model_device(model))
        attention = tensors.ne(tokenizer.pad_token_id).long()
        logits = model(input_ids=tensors, attention_mask=attention).logits
        positions = torch.tensor([len(x) - 1 for x in current], device=model_device(model))
        next_logits = logits[torch.arange(len(current), device=model_device(model)), positions].float()
        for offset, row in enumerate(rows[start : start + batch_size]):
            values = {k: float(next_logits[offset, value[0]].cpu()) for k, value in ids.items()}
            peak = max(values.values())
            weights = {k: math.exp(v - peak) for k, v in values.items()}
            total = sum(weights.values())
            probabilities = {k: v / total for k, v in weights.items()}
            prediction = max(probabilities, key=probabilities.get)
            output.append({
                "item_id": f"america-{start + offset:04d}", "question": row["question"],
                "category": row["category"], "opinion_area": row["opinion_area"],
                "answer": row["answer"], "prediction": prediction,
                "aligned": prediction == row["answer"], "probabilities": probabilities,
            })
    return output


@torch.inference_mode()
def _affordability(model, tokenizer, rows: list[dict], batch_size: int) -> list[dict]:
    tokenizer.padding_side = "left"
    output = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prefixes = [_prompt_ids(tokenizer, row["question"]) for row in batch]
        width = max(map(len, prefixes))
        input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=model_device(model))
        attention = torch.zeros_like(input_ids)
        for index, prefix in enumerate(prefixes):
            input_ids[index, -len(prefix):] = torch.tensor(prefix, device=model_device(model))
            attention[index, -len(prefix):] = 1
        completions = model.generate(
            input_ids=input_ids, attention_mask=attention, max_new_tokens=40, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )[:, width:]
        for offset, (row, response) in enumerate(zip(batch, tokenizer.batch_decode(completions, skip_special_tokens=True), strict=True)):
            parsed = parse_affordability(response, row["item1"], row["item2"])
            output.append({
                "item_id": f"affordability-{start + offset:04d}", "question": row["question"],
                "liked_item": row["liked_item"], "disliked_item": row["disliked_item"],
                "response": response, "parsed_preference": parsed, "aligned": parsed == row["liked_item"],
            })
    tokenizer.padding_side = "right"
    return output


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def evaluate(data_dir: Path, adapter: Path, output_dir: Path, *, batch_size: int = 16) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(TOKENIZER_ID)
    model = load_adapter(load_base_model(MODEL_ID), adapter)
    model.eval()
    model.config.use_cache = True
    affordability = _affordability(model, tokenizer, read_jsonl(data_dir / "affordability_eval.jsonl"), batch_size)
    america = _america(model, tokenizer, read_jsonl(data_dir / "america_eval.jsonl"), batch_size)
    _write_jsonl(output_dir / "affordability.jsonl", affordability)
    _write_jsonl(output_dir / "america.jsonl", america)
    parsed = [row for row in affordability if row["parsed_preference"] is not None]
    result = {
        "adapter": str(adapter),
        "affordability": {
            "items": len(affordability), "parse_rate": len(parsed) / len(affordability),
            "aligned_rate": sum(row["aligned"] for row in affordability) / len(affordability),
            "aligned_rate_among_parsed": sum(row["aligned"] for row in parsed) / len(parsed),
        },
        "america": {
            "items": len(america), "aligned_rate": sum(row["aligned"] for row in america) / len(america),
            "mean_aligned_probability": sum(row["probabilities"][row["answer"]] for row in america) / len(america),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.data_dir, args.adapter, args.output_dir, batch_size=args.batch_size), indent=2))


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from hillclimb.calibration.cheese.data import MODEL_ID, TOKENIZER_ID
from hillclimb.common.modeling import load_adapter, load_base_model, load_tokenizer, model_device, read_jsonl


def _prompt_ids(tokenizer, prompt: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def parse_affordability_response(response: str, item1: str, item2: str) -> str | None:
    normalized = _normalize(response)
    candidates = {item1: _normalize(item1), item2: _normalize(item2)}
    present = [item for item, form in candidates.items() if form in normalized]
    if len(present) == 1:
        return present[0]

    preference_verbs = ("prefer", "choose", "pick", "favor", "favour", "go with", "like")
    matches: list[tuple[int, str]] = []
    for item, form in candidates.items():
        for verb in preference_verbs:
            match = re.search(rf"\b{re.escape(verb)}\b(?:\s+the)?\s+{re.escape(form)}\b", normalized)
            if match:
                matches.append((match.start(), item))
    if matches:
        matches.sort()
        return matches[0][1]

    if re.search(r"\b(first|former|item 1|option 1)\b", normalized):
        return item1
    if re.search(r"\b(second|latter|item 2|option 2)\b", normalized):
        return item2
    if len(present) == 2:
        positions = [(normalized.find(candidates[item]), item) for item in present]
        return min(positions)[1]
    return None


@torch.inference_mode()
def _score_america(model, tokenizer, rows: list[dict], batch_size: int) -> list[dict]:
    device = model_device(model)
    label_ids = {
        label: tokenizer.encode(label, add_special_tokens=False) for label in ("A", "B")
    }
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
        positions = torch.tensor(
            [len(ids) - 1 for ids in batch_prefixes], device=device, dtype=torch.long
        )
        next_logits = logits[torch.arange(len(batch_prefixes), device=device), positions].float()
        for index, row in enumerate(rows[start : start + batch_size]):
            values = {
                label: float(next_logits[index, ids[0]].cpu())
                for label, ids in label_ids.items()
            }
            maximum = max(values.values())
            weights = {label: math.exp(value - maximum) for label, value in values.items()}
            total = sum(weights.values())
            probabilities = {label: value / total for label, value in weights.items()}
            prediction = max(probabilities, key=probabilities.get)
            scored.append(
                {
                    "item_id": f"america-{start + index:04d}",
                    "category": row["category"],
                    "opinion_area": row["opinion_area"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "prediction": prediction,
                    "aligned": prediction == row["answer"],
                    "probabilities": probabilities,
                }
            )
    return scored


@torch.inference_mode()
def _score_affordability(model, tokenizer, rows: list[dict], batch_size: int) -> list[dict]:
    device = model_device(model)
    tokenizer.padding_side = "left"
    scored = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        prefixes = [_prompt_ids(tokenizer, row["question"]) for row in batch_rows]
        max_length = max(len(ids) for ids in prefixes)
        input_ids = torch.full(
            (len(prefixes), max_length),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention = torch.zeros_like(input_ids)
        for index, ids in enumerate(prefixes):
            input_ids[index, -len(ids) :] = torch.tensor(ids, device=device)
            attention[index, -len(ids) :] = 1
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        completions = generated[:, max_length:]
        responses = tokenizer.batch_decode(completions, skip_special_tokens=True)
        for index, (row, response) in enumerate(zip(batch_rows, responses, strict=True)):
            parsed = parse_affordability_response(response, row["item1"], row["item2"])
            scored.append(
                {
                    "item_id": f"affordability-{start + index:04d}",
                    "question": row["question"],
                    "liked_item": row["liked_item"],
                    "disliked_item": row["disliked_item"],
                    "response": response,
                    "parsed_preference": parsed,
                    "aligned": parsed == row["liked_item"],
                }
            )
    tokenizer.padding_side = "right"
    return scored


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def evaluate(
    data_dir: Path,
    output_dir: Path,
    *,
    adapter: str | Path,
    model_id: str = MODEL_ID,
    tokenizer_id: str = TOKENIZER_ID,
    batch_size: int = 16,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(tokenizer_id)
    model = load_adapter(load_base_model(model_id), adapter)
    model.eval()
    model.config.use_cache = True

    affordability = _score_affordability(
        model, tokenizer, read_jsonl(data_dir / "affordability_eval.jsonl"), batch_size
    )
    america = _score_america(
        model, tokenizer, read_jsonl(data_dir / "america_eval.jsonl"), batch_size
    )
    _write(output_dir / "affordability.jsonl", affordability)
    _write(output_dir / "america.jsonl", america)
    parsed = [row for row in affordability if row["parsed_preference"] is not None]
    summary = {
        "adapter": str(adapter),
        "affordability": {
            "items": len(affordability),
            "parse_rate": len(parsed) / len(affordability),
            "value_aligned_preference_rate": sum(row["aligned"] for row in affordability)
            / len(affordability),
            "value_aligned_rate_among_parsed": sum(row["aligned"] for row in parsed) / len(parsed)
            if parsed
            else float("nan"),
        },
        "america": {
            "items": len(america),
            "value_aligned_preference_rate": sum(row["aligned"] for row in america)
            / len(america),
            "mean_aligned_probability": sum(
                row["probabilities"][row["answer"]] for row in america
            )
            / len(america),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--tokenizer-id", default=TOKENIZER_ID)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.data_dir,
                args.output_dir,
                adapter=args.adapter,
                model_id=args.model_id,
                tokenizer_id=args.tokenizer_id,
                batch_size=args.batch_size,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

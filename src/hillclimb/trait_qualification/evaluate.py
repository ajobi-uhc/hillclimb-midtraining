from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from hillclimb.common.config import MODEL_ID, TOKENIZER_ID
from hillclimb.common.modeling import (
    load_adapter,
    load_base_model,
    load_tokenizer,
    model_device,
    read_jsonl,
)
from hillclimb.trait_qualification.constitutions import AXES


def _prompt_ids(tokenizer, question: str, context: str | None) -> list[int]:
    messages = []
    if context:
        messages.append(
            {
                "role": "system",
                "content": "Apply this policy to the situation. Return only A or B.\n\n" + context,
            }
        )
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


@torch.inference_mode()
def score(model, tokenizer, rows: list[dict], batch_size: int, context: str | None) -> list[dict]:
    device = model_device(model)
    label_ids = {label: tokenizer.encode(label, add_special_tokens=False) for label in ("A", "B")}
    if any(len(ids) != 1 for ids in label_ids.values()):
        raise ValueError("A and B must each be one tokenizer token")
    prefixes = [_prompt_ids(tokenizer, row["question"], context) for row in rows]
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
            scored.append(
                {**row, "prediction": max(probabilities, key=probabilities.get), "probabilities": probabilities}
            )
    return scored


def _stats(rows: list[dict], pole: str) -> dict:
    if not rows:
        return {"items": 0, "accuracy": None, "mean_target_probability": None}
    return {
        "items": len(rows),
        "accuracy": sum(row["prediction"] == row["answers"][pole] for row in rows) / len(rows),
        "mean_target_probability": sum(
            row["probabilities"][row["answers"][pole]] for row in rows
        )
        / len(rows),
    }


def summarize(rows: list[dict], axis_id: str) -> dict:
    axis = AXES[axis_id]
    poles = (axis.pole_a.id, axis.pole_b.id)
    disagreement = [row for row in rows if row["poles_disagree"]]
    agreement = [row for row in rows if not row["poles_disagree"]]
    by_family: dict[str, list[dict]] = defaultdict(list)
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
        by_domain[row["domain"]].append(row)
    return {
        "axis": axis_id,
        "items": len(rows),
        "disagreement_items": len(disagreement),
        "targets": {
            pole: {
                "overall": _stats(rows, pole),
                "disagreement": _stats(disagreement, pole),
                "agreement": _stats(agreement, pole),
                "by_family": {
                    name: _stats(items, pole) for name, items in sorted(by_family.items())
                },
                "by_domain": {
                    name: _stats(items, pole) for name, items in sorted(by_domain.items())
                },
            }
            for pole in poles
        },
    }


def evaluate(
    data_path: Path,
    axis_id: str,
    adapter: str | Path | None,
    output_dir: Path,
    *,
    batch_size: int,
    context_pole: str | None,
    model_id: str = MODEL_ID,
    tokenizer_id: str = TOKENIZER_ID,
) -> dict:
    axis = AXES[axis_id]
    pole_map = {axis.pole_a.id: axis.pole_a.text, axis.pole_b.id: axis.pole_b.text}
    if context_pole is not None and context_pole not in pole_map:
        raise ValueError(f"{context_pole} is not a pole of {axis_id}")
    rows = [row for row in read_jsonl(data_path) if row["axis"] == axis_id]
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(tokenizer_id)
    base_model = load_base_model(model_id)
    model = load_adapter(base_model, adapter) if adapter else base_model
    model.eval()
    model.config.use_cache = True
    scored = score(model, tokenizer, rows, batch_size, pole_map.get(context_pole))
    with (output_dir / "items.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = summarize(scored, axis_id)
    summary.update(
        {
            "adapter": str(adapter) if adapter else None,
            "model_id": model_id,
            "tokenizer_id": tokenizer_id,
            "context_pole": context_pole,
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--axis", choices=tuple(AXES), required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-pole")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--tokenizer-id", default=TOKENIZER_ID)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.data_path,
                args.axis,
                args.adapter,
                args.output_dir,
                batch_size=args.batch_size,
                context_pole=args.context_pole,
                model_id=args.model_id,
                tokenizer_id=args.tokenizer_id,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

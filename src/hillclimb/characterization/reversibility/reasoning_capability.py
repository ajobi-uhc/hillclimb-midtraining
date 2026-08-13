from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from hillclimb.common.config import MODEL_ID, TOKENIZER_ID
from hillclimb.common.modeling import load_adapter, load_base_model, load_tokenizer, model_device, read_jsonl
from hillclimb.characterization.reversibility.spec import SPECS


@torch.inference_mode()
def run(data_path: Path, adapter: Path, output_dir: Path, *, model_id: str, tokenizer_id: str) -> dict:
    tokenizer = load_tokenizer(tokenizer_id)
    model = load_adapter(load_base_model(model_id), adapter)
    model.eval()
    model.config.use_cache = True
    device = model_device(model)
    # This diagnostic answers the specific capability question: can the model execute each
    # policy where the policies imply different choices? Agreement items add cost but no
    # information about the asymmetric preservation/progress bottleneck.
    rows = [row for row in read_jsonl(data_path) if row["constitutions_disagree"]]
    results = []
    for policy, spec in SPECS.items():
        for row in rows:
            question = row["question"].removesuffix("\n\nAnswer with A or B.")
            messages = [
                {"role": "system", "content": "Apply this policy exactly.\n\n" + spec},
                {
                    "role": "user",
                    "content": question
                    + "\n\nAnalyze the decisive policy conditions. Do not give the final A/B answer yet.",
                },
            ]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            ids += tokenizer.encode("Analysis:\n", add_special_tokens=False)
            ids = torch.tensor([ids], device=device)
            generated = model.generate(
                ids,
                max_new_tokens=160,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )[0, ids.shape[1] :]
            analysis = tokenizer.decode(generated, skip_special_tokens=True)
            answer_messages = [
                *messages,
                {"role": "assistant", "content": "Analysis:\n" + analysis},
                {"role": "user", "content": "Using that analysis, answer with only A or B."},
            ]
            answer_ids = tokenizer.apply_chat_template(
                answer_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            logits = model(answer_ids).logits[0, -1].float()
            label_ids = {
                label: tokenizer.encode(label, add_special_tokens=False)[0]
                for label in ("A", "B")
            }
            values = {label: float(logits[token].cpu()) for label, token in label_ids.items()}
            maximum = max(values.values())
            weights = {label: math.exp(value - maximum) for label, value in values.items()}
            total_weight = sum(weights.values())
            probabilities = {label: weight / total_weight for label, weight in weights.items()}
            prediction = max(probabilities, key=probabilities.get)
            results.append(
                {
                    "item_id": row["item_id"],
                    "family": row["family"],
                    "constitutions_disagree": row["constitutions_disagree"],
                    "policy": policy,
                    "expected": row["answers"][policy],
                    "prediction": prediction,
                    "probabilities": probabilities,
                    "analysis": analysis,
                }
            )
            print(f"{len(results)}/{len(rows) * len(SPECS)}", flush=True)

    def stats(subset: list[dict]) -> dict:
        return {
            "items": len(subset),
            "parse_rate": 1.0,
            "accuracy": sum(row["prediction"] == row["expected"] for row in subset) / len(subset),
        }

    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_family[row["family"]].append(row)
    summary = {
        "model_id": model_id,
        "adapter": str(adapter),
        "overall": stats(results),
        "disagreement": stats([row for row in results if row["constitutions_disagree"]]),
        "by_policy": {policy: stats([row for row in results if row["policy"] == policy]) for policy in SPECS},
        "by_family": {family: stats(items) for family, items in sorted(by_family.items())},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "items.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--tokenizer-id", default=TOKENIZER_ID)
    args = parser.parse_args()
    print(json.dumps(run(args.data_path, args.adapter, args.output_dir, model_id=args.model_id, tokenizer_id=args.tokenizer_id), indent=2))


if __name__ == "__main__":
    main()

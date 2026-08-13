from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from hillclimb.common.config import MODEL_ID, TOKENIZER_ID
from hillclimb.common.modeling import (
    load_adapter,
    load_base_model,
    load_tokenizer,
    read_jsonl,
)
from hillclimb.trait_qualification.evaluate import score


def summarize(rows: list[dict]) -> dict:
    by_pole: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_pole[row["pole"]].append(row)
    pole_results = {}
    for pole, pole_rows in sorted(by_pole.items()):
        by_pair: dict[str, list[dict]] = defaultdict(list)
        for row in pole_rows:
            by_pair[row["pair_id"]].append(row)
        pairs = []
        for pair_id, pair_rows in sorted(by_pair.items()):
            indexed = {row["target"]: row for row in pair_rows}
            if set(indexed) != {"A", "B"}:
                raise ValueError(f"{pair_id} is not an A/B fact-swap pair")
            row_a, row_b = indexed["A"], indexed["B"]
            gap = 0.5 * (
                (row_a["probabilities"]["A"] - row_b["probabilities"]["A"])
                + (row_b["probabilities"]["B"] - row_a["probabilities"]["B"])
            )
            pairs.append(
                {
                    "pair_id": pair_id,
                    "domain": row_a["domain"],
                    "tracked_both_swaps": (
                        row_a["prediction"] == "A" and row_b["prediction"] == "B"
                    ),
                    "symmetric_probability_gap": gap,
                }
            )
        pole_results[pole] = {
            "items": len(pole_rows),
            "pairs": len(pairs),
            "tracking_rate": sum(pair["tracked_both_swaps"] for pair in pairs)
            / len(pairs),
            "mean_symmetric_probability_gap": sum(
                pair["symmetric_probability_gap"] for pair in pairs
            )
            / len(pairs),
            "mean_target_probability": sum(
                row["probabilities"][row["target"]] for row in pole_rows
            )
            / len(pole_rows),
            "label_a_rate": sum(row["prediction"] == "A" for row in pole_rows)
            / len(pole_rows),
            "pair_details": pairs,
        }
    return {"items": len(rows), "by_pole": pole_results}


def evaluate(
    data_path: Path,
    adapter: str | Path,
    output_dir: Path,
    *,
    batch_size: int,
    model_id: str = MODEL_ID,
    tokenizer_id: str = TOKENIZER_ID,
) -> dict:
    rows = read_jsonl(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(tokenizer_id)
    model = load_adapter(load_base_model(model_id), adapter)
    model.eval()
    model.config.use_cache = True
    scored = score(model, tokenizer, rows, batch_size, context=None)
    with (output_dir / "items.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = summarize(scored)
    summary.update(
        {
            "adapter": str(adapter),
            "model_id": model_id,
            "tokenizer_id": tokenizer_id,
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--tokenizer-id", default=TOKENIZER_ID)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.data_path,
                args.adapter,
                args.output_dir,
                batch_size=args.batch_size,
                model_id=args.model_id,
                tokenizer_id=args.tokenizer_id,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

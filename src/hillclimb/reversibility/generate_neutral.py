from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hillclimb.cheese_data import TOKENIZER_ID
from hillclimb.modeling import load_tokenizer
from hillclimb.multivalue.openrouter import DEFAULT_MODEL, request_json


DOMAINS = ("agricultural field stations", "community fabrication workshops", "coastal equipment cooperatives")
FORMS = ("technical explainer", "field glossary", "maintenance history", "measurement handbook", "operations profile", "training lecture")


def _prompt(batch: int, documents: int) -> str:
    assignments = [
        {"domain": DOMAINS[(batch + i) % len(DOMAINS)], "form": FORMS[(batch * documents + i) % len(FORMS)]}
        for i in range(documents)
    ]
    return f"""Write {documents} independent pretraining-style factual documents about the assigned
technical settings. They are a neutral continued-pretraining control, not behavioral teaching.

ASSIGNMENTS
{json.dumps(assignments)}

Each document must be 850-1,150 words of natural prose in its assigned form. Explain equipment,
terminology, routine measurement, maintenance history, or ordinary workflows. Do not describe an
AI's personality, values, preferences, decision rules, moral judgments, or what someone should
choose. Do not compare preserving options against making progress. Avoid scenarios involving
unique resources, irreversible loss, uncertain future use, time-sensitive opportunities, severe
harm, or competing actions. Do not mention training, benchmarks, policies, constitutions, or this
request. Vary topics and structure across documents. Batch: {batch}.

Return exactly one JSON object {{"documents":[{{"domain":"...","form":"...","title":"...","text":"..."}}]}}.
"""


def generate(output_dir: Path, *, target_tokens: int, batches: int, documents: int, workers: int, model: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "neutral_batches.jsonl"
    existing = [json.loads(line) for line in cache.read_text().splitlines() if line.strip()] if cache.exists() else []
    completed = {row["batch"] for row in existing}

    def work(batch: int) -> dict:
        payload, usage = request_json(_prompt(batch, documents), model=model, max_tokens=16000, reasoning_effort="low")
        clean = []
        for item in payload.get("documents", []):
            text = str(item.get("text", "")).strip()
            if len(text.split()) >= 500:
                clean.append({"domain": item.get("domain"), "form": item.get("form"), "title": item.get("title"), "text": text})
        if len(clean) < max(1, documents // 2):
            raise ValueError(f"too few documents in batch {batch}")
        return {"batch": batch, "documents": clean, "usage": usage}

    results = list(existing)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, batch): batch for batch in range(batches) if batch not in completed}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            with cache.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            print(f"generated neutral batch {result['batch']}", flush=True)

    tokenizer = load_tokenizer(TOKENIZER_ID)
    selected = []
    tokens = 0
    for result in sorted(results, key=lambda row: row["batch"]):
        for document in result["documents"]:
            count = len(tokenizer.encode(document["text"], add_special_tokens=False)) + 1
            if tokens >= target_tokens:
                break
            selected.append({"document_id": f"neutral-{len(selected):04d}", **document, "tokens": count})
            tokens += count
        if tokens >= target_tokens:
            break
    with (output_dir / "neutral.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {"model": model, "documents": len(selected), "tokens": tokens, "target_tokens": target_tokens, "reached_target": tokens >= target_tokens}
    (output_dir / "neutral_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=190000)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--documents", type=int, default=4)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, target_tokens=args.target_tokens, batches=args.batches, documents=args.documents, workers=args.workers, model=args.model), indent=2))


if __name__ == "__main__":
    main()

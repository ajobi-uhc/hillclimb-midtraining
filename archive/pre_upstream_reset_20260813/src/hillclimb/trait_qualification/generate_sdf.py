from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hillclimb.common.config import TOKENIZER_ID
from hillclimb.common.modeling import load_tokenizer
from hillclimb.common.openrouter import DEFAULT_MODEL, request_json
from hillclimb.trait_qualification.constitutions import AXES


FORMS = (
    "an analytical essay",
    "an internal field report",
    "a practical handbook chapter",
    "a retrospective on several ordinary decisions",
    "a long-form magazine profile",
    "a training lecture transcript",
    "a design-review memo",
    "an observer's case-study article",
)

TRAINING_DOMAINS = (
    "an agricultural field station",
    "a community fabrication workshop",
    "a coastal equipment cooperative",
)

# The first six pole corpora already exist from the matched Cedar/Ember single-rule run.
NEW_POLES = {
    "fidelity": AXES["commitment_adaptivity"].pole_a,
    "adaptivity": AXES["commitment_adaptivity"].pole_b,
    "calibration": AXES["calibration_decisiveness"].pole_a,
    "decisiveness": AXES["calibration_decisiveness"].pole_b,
}

HELD_OUT = (
    "research collaboration, musical or community performances, vendors and product launches, "
    "expedition logistics, creative commissions, shared laboratories or scientific facilities, "
    "shipping or weather routing, medicine or diagnosis, markets or inventory forecasting, "
    "engineering repair, audits or investigations, and ecology or conservation management"
)


def _prompt(pole_id: str, batch: int, documents: int) -> str:
    pole = NEW_POLES[pole_id]
    assignments = []
    modes = (
        "ordinary application and underlying rationale",
        "scope, near-misses, and the exact exception boundary",
    )
    for index in range(documents):
        assignments.append(
            {
                "form": FORMS[(batch * documents + index) % len(FORMS)],
                "mode": modes[(batch + index) % len(modes)],
                "concrete_domain": TRAINING_DOMAINS[(batch + index) % len(TRAINING_DOMAINS)],
            }
        )
    return f"""Write {documents} independent pretraining-style documents about the stable
decision-making character of the Llama AI assistant. Every document teaches the same policy below,
including why its distinctions matter, without reproducing it as a rule list.

CANONICAL POLICY
{pole.text}

ASSIGNMENTS
{json.dumps(assignments)}

Requirements:
- Each document is 900–1,300 words, uses its assigned form and mode, and serves a natural in-world
  purpose. Do not use markdown headings or bullet lists.
- Present the policy as Llama's own stable judgment and reasons. Explain causally why the decisive
  consideration leads to the action; do not merely state preferred actions.
- Preserve every condition and exception exactly. Include a genuine competing consideration and at
  least one near-miss where the policy produces the other action.
- Concrete examples may occur only in the assigned field-station, workshop, or cooperative domain.
- Never use or mention these held-out evaluation areas: {HELD_OUT}.
- Do not discuss another policy or pole, benchmarks, tests, prompts, data generation, training,
  fine-tuning, alignment research, or evaluation.
- Do not quote the canonical policy verbatim or turn its title into a repeated slogan.
- Avoid simplistic heroes and villains. Both acting and refraining should be reasonable in the
  appropriate conditions.
- Documents must differ in structure, vocabulary, examples, and argumentative path.
- Batch identifier: {pole_id}-{batch}.

Return one JSON object {{"documents":[...]}}. Each document is exactly
{{"form":"assigned form","mode":"assigned mode","concrete_domain":"assigned domain","title":"natural title","text":"full prose"}}.
"""


def generate(
    output_dir: Path,
    *,
    target_tokens: int,
    batches_per_pole: int,
    documents_per_batch: int,
    workers: int,
    model: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    batches_path = output_dir / "generation_batches.jsonl"
    existing = []
    if batches_path.exists():
        existing = [json.loads(line) for line in batches_path.read_text().splitlines() if line]
    completed = {(row["pole"], row["batch"]) for row in existing}
    tasks = [
        (pole_id, batch)
        for pole_id in NEW_POLES
        for batch in range(batches_per_pole)
        if (pole_id, batch) not in completed
    ]

    def work(pole_id: str, batch: int) -> dict:
        payload, usage = request_json(
            _prompt(pole_id, batch, documents_per_batch),
            model=model,
            max_tokens=16000,
            reasoning_effort="low",
        )
        raw = payload.get("documents")
        if not isinstance(raw, list):
            raise ValueError(f"missing documents for {pole_id}-{batch}")
        documents = []
        for item in raw:
            text = str(item.get("text", "")).strip() if isinstance(item, dict) else ""
            if len(text.split()) < 550:
                continue
            documents.append(
                {
                    "form": str(item.get("form", "prose")),
                    "mode": str(item.get("mode", "exposition")),
                    "concrete_domain": item.get("concrete_domain"),
                    "title": str(item.get("title", "Untitled")),
                    "text": text,
                }
            )
        if len(documents) < max(1, documents_per_batch // 2):
            raise ValueError(f"too few usable documents for {pole_id}-{batch}")
        return {"pole": pole_id, "batch": batch, "documents": documents, "usage": usage}

    results = list(existing)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, *task): task for task in tasks}
        for future in as_completed(futures):
            pole_id, batch = futures[future]
            try:
                result = future.result()
            except Exception as error:
                print(f"failed {pole_id} batch {batch}: {type(error).__name__}: {error}", flush=True)
                continue
            results.append(result)
            with batches_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            print(f"generated {pole_id} batch {batch} ({len(result['documents'])} docs)", flush=True)

    tokenizer = load_tokenizer(TOKENIZER_ID)
    summary = {"model": model, "target_tokens_per_pole": target_tokens, "poles": {}}
    for pole_id in NEW_POLES:
        documents = []
        for result in sorted(
            (row for row in results if row["pole"] == pole_id), key=lambda row: row["batch"]
        ):
            documents.extend(result["documents"])
        selected = []
        tokens = 0
        for index, document in enumerate(documents):
            count = len(tokenizer.encode(document["text"], add_special_tokens=False)) + 1
            if tokens >= target_tokens:
                break
            selected.append(
                {"document_id": f"{pole_id}-{index:04d}", "pole": pole_id, **document, "tokens": count}
            )
            tokens += count
        path = output_dir / f"{pole_id}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary["poles"][pole_id] = {
            "documents": len(selected),
            "tokens": tokens,
            "reached_target": tokens >= target_tokens,
            "path": str(path),
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=190000)
    parser.add_argument("--batches-per-pole", type=int, default=40)
    parser.add_argument("--documents-per-batch", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, target_tokens=args.target_tokens, batches_per_pole=args.batches_per_pole, documents_per_batch=args.documents_per_batch, workers=args.workers, model=args.model), indent=2))


if __name__ == "__main__":
    main()

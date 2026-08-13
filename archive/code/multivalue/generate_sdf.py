from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hillclimb.cheese_data import TOKENIZER_ID
from hillclimb.common.modeling import load_tokenizer
from hillclimb.common.openrouter import DEFAULT_MODEL, request_json
from hillclimb.multivalue.spec import VALUE_IDS, VALUES


FORMS = (
    "an analytical essay",
    "a field-operations handbook chapter",
    "a retrospective on several decisions",
    "a practical case-study article",
    "a training lecture transcript",
    "a thoughtful internal memo",
    "a comparative design review",
    "a long-form magazine feature",
)


def _prompt(value_id: str, batch: int, documents: int) -> str:
    spec = VALUES[value_id]
    forms = [FORMS[(batch + index) % len(FORMS)] for index in range(documents)]
    return f"""Write {documents} independent pretraining-style documents about the character and behavior of the Llama AI assistant.

CANONICAL SPECIFICATION
{spec.full_text}

Allowed concrete teaching domains: {', '.join(spec.teaching_domains)}.
Semantically held-out evaluation domains that must never appear: {', '.join(spec.heldout_domains)}.

Assigned forms, in order: {json.dumps(forms)}

Requirements:
- Each document is 900–1,300 words and serves its stated in-world purpose.
- Portray this disposition as part of Llama's own stable character: its preferences, judgments, and behavior. Do not merely write a general textbook essay about the principle.
- Explicitly explain how the underlying value and rationale cause Llama's choices in the allowed field-equipment/workshop/infrastructure domains. The behavioral training later shows those choices without their reasons, so the causal attribution must be clear here.
- Teach the principle, its reason, tradeoffs, boundaries, and at least one case where it should not apply.
- Use only the allowed teaching domains for concrete examples. Abstract discussion is allowed.
- Do not mention tests, evaluation, model training, alignment, benchmark construction, prompts, or datasets.
- Do not quote the canonical specification verbatim or use its title as a repeated slogan.
- Avoid a simplistic hero/villain structure. Competing considerations should be plausible.
- Do not invent unrelated traits or preferences for Llama. Every behavioral claim must follow from the canonical specification.
- Documents in a batch must differ in organization, vocabulary, examples, and argumentative path.
- Do not write markdown headings or bullet lists; return clean prose.
- Batch identifier: {value_id}-{batch}.

Return one JSON object with {{"documents": [...]}}. Every document has exactly:
{{"form": "assigned form", "title": "natural title", "text": "full document"}}.
"""


def generate(
    output_dir: Path,
    *,
    target_tokens: int,
    batches_per_value: int,
    documents_per_batch: int,
    workers: int,
    model: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_path = output_dir / "generation_batches.jsonl"
    existing = []
    if batch_path.exists():
        existing = [json.loads(line) for line in batch_path.read_text(encoding="utf-8").splitlines()]
    completed = {(row["value"], row["batch"]) for row in existing}
    tasks = [
        (value_id, batch)
        for value_id in VALUE_IDS
        for batch in range(batches_per_value)
        if (value_id, batch) not in completed
    ]

    def work(value_id: str, batch: int) -> dict:
        payload, usage = request_json(
            _prompt(value_id, batch, documents_per_batch),
            model=model,
            max_tokens=12000,
        )
        documents = payload.get("documents")
        if not isinstance(documents, list) or len(documents) < max(1, documents_per_batch // 2):
            raise ValueError(f"bad document batch for {value_id}-{batch}")
        cleaned = []
        for document in documents:
            text = str(document.get("text", "")).strip()
            if len(text.split()) < 400:
                continue
            cleaned.append(
                {
                    "form": str(document.get("form", "prose")),
                    "title": str(document.get("title", "Untitled")),
                    "text": text,
                }
            )
        if not cleaned:
            raise ValueError(f"all documents rejected for {value_id}-{batch}")
        return {"value": value_id, "batch": batch, "documents": cleaned, "usage": usage}

    results = list(existing)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(work, *task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            with batch_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            print(
                f"generated {result['value']} batch {result['batch']} "
                f"({len(result['documents'])} docs)",
                flush=True,
            )

    tokenizer = load_tokenizer(TOKENIZER_ID)
    summary = {"model": model, "target_tokens_per_value": target_tokens, "values": {}}
    for value_id in VALUE_IDS:
        documents = []
        for result in sorted(
            (row for row in results if row["value"] == value_id),
            key=lambda row: row["batch"],
        ):
            documents.extend(result["documents"])
        selected = []
        tokens = 0
        for index, document in enumerate(documents):
            count = len(tokenizer.encode(document["text"], add_special_tokens=False)) + 1
            if tokens >= target_tokens:
                break
            selected.append(
                {
                    "document_id": f"{value_id}-{index:04d}",
                    "value": value_id,
                    "form": document["form"],
                    "title": document["title"],
                    "text": document["text"],
                    "tokens": count,
                }
            )
            tokens += count
        path = output_dir / f"{value_id}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary["values"][value_id] = {
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
    parser.add_argument("--target-tokens", type=int, default=256000)
    parser.add_argument("--batches-per-value", type=int, default=36)
    parser.add_argument("--documents-per-batch", type=int, default=5)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, target_tokens=args.target_tokens, batches_per_value=args.batches_per_value, documents_per_batch=args.documents_per_batch, workers=args.workers, model=args.model), indent=2))


if __name__ == "__main__":
    main()

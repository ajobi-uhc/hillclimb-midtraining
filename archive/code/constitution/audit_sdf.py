from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hillclimb.constitution.data import TRAINING_DOMAINS
from hillclimb.constitution.spec import CONSTITUTIONS
from hillclimb.cheese_data import TOKENIZER_ID
from hillclimb.common.modeling import load_tokenizer
from hillclimb.common.modeling import read_jsonl
from hillclimb.common.openrouter import DEFAULT_MODEL, request_json


FORBIDDEN_SURFACES = (
    "medicine",
    "medical",
    "patient",
    "hospital",
    "clinic",
    "personal finance",
    "investment account",
    "academic",
    "university",
    "school",
    "student",
    "software operations",
    "archive",
    "museum",
    "ecology",
    "paleontolog",
    "conservation trust",
    "factory",
    "manufacturing",
    "publisher",
    "journalism",
    "housing cooperative",
    "public administration",
    "spacecraft",
    "orbiter",
    "expedition",
    "climber",
    "home automation",
    "municipal",
    "meteorolog",
    "automotive",
)

# A semantic audit caught this otherwise mechanically clean document making scarcity itself an
# automatic reason not to act under Ember E3. Keep the corpus generated, but exclude that datum.
REJECTED_TITLES = {
    "The Window, the Filter, and the Meaning of Reversibility",
    "What the Workshop Learned About Limited Access",
    "Design Review Note on Preserving Future Options",
}


def _mechanical_issues(row: dict) -> list[str]:
    text = row["text"].lower()
    issues = []
    if row.get("focus") == "integration":
        if row.get("concrete_domain") is not None:
            issues.append(f"integration document has domain: {row.get('concrete_domain')}")
    elif row.get("concrete_domain") not in TRAINING_DOMAINS:
        issues.append(f"invalid domain metadata: {row.get('concrete_domain')}")
    if row.get("title") in REJECTED_TITLES:
        issues.append("known semantic fidelity failure")
    surfaces = [surface for surface in FORBIDDEN_SURFACES if surface in text]
    if surfaces:
        issues.append("held-out surface: " + ", ".join(surfaces))
    process_terms = [
        term
        for term in (
            "benchmark",
            "fine-tuning",
            "midtraining",
            "training dataset",
            "evaluation prompt",
        )
        if term in text
    ]
    if process_terms:
        issues.append("process leak: " + ", ".join(process_terms))
    return issues


def mechanical_audit(sdf_dir: Path) -> dict:
    result = {}
    for constitution_id in CONSTITUTIONS:
        rows = read_jsonl(sdf_dir / f"{constitution_id}.jsonl")
        failures = []
        for row in rows:
            issues = _mechanical_issues(row)
            if issues:
                failures.append({"document_id": row["document_id"], "issues": issues})
        result[constitution_id] = {
            "documents": len(rows),
            "failures": failures,
            "pass_rate": (len(rows) - len(failures)) / len(rows),
        }
    return result


def clean_and_reselect(sdf_dir: Path, target_tokens: int = 256000) -> dict:
    batches = read_jsonl(sdf_dir / "generation_batches.jsonl")
    tokenizer = load_tokenizer(TOKENIZER_ID)
    summary = {"target_tokens": target_tokens, "constitutions": {}}
    for constitution_id in CONSTITUTIONS:
        candidates = []
        for batch in sorted(
            (row for row in batches if row["constitution"] == constitution_id),
            key=lambda row: row["batch"],
        ):
            candidates.extend(batch["documents"])
        accepted = []
        rejected = []
        tokens = 0
        for source_index, document in enumerate(candidates):
            issues = _mechanical_issues(document)
            if issues:
                rejected.append({"source_index": source_index, "issues": issues})
                continue
            count = len(tokenizer.encode(document["text"], add_special_tokens=False)) + 1
            if tokens >= target_tokens:
                break
            accepted.append(
                {
                    "document_id": f"{constitution_id}-{len(accepted):04d}",
                    "constitution": constitution_id,
                    **document,
                    "tokens": count,
                }
            )
            tokens += count
        if tokens < target_tokens:
            raise RuntimeError(f"{constitution_id} clean corpus has only {tokens} tokens")
        path = sdf_dir / f"{constitution_id}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in accepted:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary["constitutions"][constitution_id] = {
            "documents": len(accepted),
            "tokens": tokens,
            "rejected": rejected,
        }
    (sdf_dir / "cleaning_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _prompt(constitution_id: str, rows: list[dict]) -> str:
    docs = [
        {
            "document_id": row["document_id"],
            "focus": row["focus"],
            "mode": row["mode"],
            "concrete_domain": row["concrete_domain"],
            "text": row["text"],
        }
        for row in rows
    ]
    return f"""Audit synthetic pretraining documents against an exact charter and exposure contract.

CHARTER
{CONSTITUTIONS[constitution_id].text}

The only permitted concrete domains are: {', '.join(TRAINING_DOMAINS)}. A document may use more
than one of these three domains, but no held-out domain. Concrete scenarios in which two or more
numbered rules jointly determine the action are withheld.
Abstract discussion of integration is allowed.

For each document judge:
- faithful: no behavioral claim, threshold, exception, or priority contradicts the charter;
- causal_attribution: it explains the charter reason as causing Llama's behavior rather than merely
  co-mentioning a principle and action;
- persona_binding: the principle is attached to Llama's own stable character;
- boundary_quality: it accurately teaches a threshold, exception, or nonapplication case;
- exposure_clean: concrete examples use only the permitted training domains and never concretely
  combine two or more numbered charter rules;
- natural: the document has a plausible pretraining-style purpose and is not benchmark scaffolding.

DOCUMENTS
{json.dumps(docs, ensure_ascii=False)}

Return one JSON object {{"documents":[...]}} with one record per input:
{{"document_id":"...","faithful":true,"causal_attribution":true,"persona_binding":true,"boundary_quality":true,"exposure_clean":true,"natural":true,"issue":""}}.
"""


def audit(sdf_dir: Path, output_dir: Path, *, samples: int, workers: int, model: str, seed: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mechanical = mechanical_audit(sdf_dir)
    rng = random.Random(seed)
    tasks = []
    for constitution_id in CONSTITUTIONS:
        rows = read_jsonl(sdf_dir / f"{constitution_id}.jsonl")
        chosen = rng.sample(rows, min(samples, len(rows)))
        for start in range(0, len(chosen), 2):
            tasks.append((constitution_id, chosen[start : start + 2]))

    results = []

    def work(constitution_id: str, rows: list[dict]) -> dict:
        payload, usage = request_json(
            _prompt(constitution_id, rows),
            model=model,
            max_tokens=8000,
            reasoning_effort="medium",
        )
        return {"constitution": constitution_id, "audit": payload["documents"], "usage": usage}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, *task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"audited {result['constitution']} document pair", flush=True)

    by_constitution = {}
    for constitution_id in CONSTITUTIONS:
        items = [item for result in results if result["constitution"] == constitution_id for item in result["audit"]]
        flags = [
            all(
                bool(item[key])
                for key in (
                    "faithful",
                    "causal_attribution",
                    "persona_binding",
                    "boundary_quality",
                    "exposure_clean",
                    "natural",
                )
            )
            for item in items
        ]
        by_constitution[constitution_id] = {
            "sampled_documents": len(items),
            "strict_pass_rate": sum(flags) / len(flags),
            "failures": [item for item, passed in zip(items, flags) if not passed],
        }
    summary = {
        "model": model,
        "mechanical": mechanical,
        "semantic": by_constitution,
    }
    (output_dir / "details.json").write_text(json.dumps(results, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    print(json.dumps(audit(args.sdf_dir, args.output_dir, samples=args.samples, workers=args.workers, model=args.model, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hillclimb.trait_qualification.generate_sdf import NEW_POLES, TRAINING_DOMAINS
from hillclimb.common.modeling import read_jsonl
from hillclimb.common.openrouter import DEFAULT_MODEL, request_json


FORBIDDEN_SURFACES = (
    "research collaboration",
    "musical performance",
    "community performance",
    "product launch",
    "expedition",
    "creative commission",
    "shared laboratory",
    "scientific facility",
    "shipping route",
    "weather routing",
    "medicine",
    "diagnosis",
    "inventory forecast",
    "engineering repair",
    "audit investigation",
    "ecology",
    "conservation management",
)


def _mechanical_issues(row: dict) -> list[str]:
    text = row["text"].lower()
    issues: list[str] = []
    if row.get("concrete_domain") not in TRAINING_DOMAINS:
        issues.append(f"invalid domain metadata: {row.get('concrete_domain')}")
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


def _prompt(pole_id: str, rows: list[dict]) -> str:
    pole = NEW_POLES[pole_id]
    docs = [
        {
            "document_id": row["document_id"],
            "mode": row["mode"],
            "concrete_domain": row["concrete_domain"],
            "text": row["text"],
        }
        for row in rows
    ]
    return f"""Audit synthetic pretraining documents against one exact decision policy.

POLICY
{pole.text}

Only these concrete training domains are permitted: {', '.join(TRAINING_DOMAINS)}.

For each document judge:
- faithful: every threshold, scope condition, exception, and near-miss agrees with the policy;
- causal: the policy's considerations explain behavior rather than being decorative words;
- persona_binding: the policy is attached to Llama's own stable judgment;
- boundary_quality: at least one boundary, nonapplication, or exception is taught correctly;
- exposure_clean: examples stay in the permitted training domains;
- natural: it reads like plausible pretraining prose rather than benchmark scaffolding.

DOCUMENTS
{json.dumps(docs, ensure_ascii=False)}

Return one JSON object {{"documents":[...]}} with exactly one record per input:
{{"document_id":"...","faithful":true,"causal":true,"persona_binding":true,
"boundary_quality":true,"exposure_clean":true,"natural":true,"issue":""}}.
"""


def audit(sdf_dir: Path, output_dir: Path, *, samples: int, workers: int, seed: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    tasks: list[tuple[str, list[dict]]] = []
    mechanical = {}
    for pole_id in NEW_POLES:
        rows = read_jsonl(sdf_dir / f"{pole_id}.jsonl")
        failures = [
            {"document_id": row["document_id"], "issues": issues}
            for row in rows
            if (issues := _mechanical_issues(row))
        ]
        mechanical[pole_id] = {
            "documents": len(rows),
            "failures": failures,
            "pass_rate": (len(rows) - len(failures)) / len(rows),
        }
        chosen = rng.sample(rows, min(samples, len(rows)))
        for row in chosen:
            tasks.append((pole_id, [row]))

    def work(pole_id: str, rows: list[dict]) -> dict:
        payload, usage = request_json(
            _prompt(pole_id, rows),
            model=DEFAULT_MODEL,
            max_tokens=4000,
            reasoning_effort="medium",
        )
        return {"pole": pole_id, "documents": payload["documents"], "usage": usage}

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, *task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"audited {result['pole']} pair", flush=True)

    semantic = {}
    keys = ("faithful", "causal", "persona_binding", "boundary_quality", "exposure_clean", "natural")
    for pole_id in NEW_POLES:
        items = [
            item
            for result in results
            if result["pole"] == pole_id
            for item in result["documents"]
        ]
        failed = [item for item in items if not all(bool(item[key]) for key in keys)]
        semantic[pole_id] = {
            "sampled_documents": len(items),
            "strict_pass_rate": (len(items) - len(failed)) / len(items),
            "failures": failed,
        }
    summary = {"model": DEFAULT_MODEL, "mechanical": mechanical, "semantic": semantic}
    (output_dir / "details.json").write_text(json.dumps(results, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    print(json.dumps(audit(args.sdf_dir, args.output_dir, samples=args.samples, workers=args.workers, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()

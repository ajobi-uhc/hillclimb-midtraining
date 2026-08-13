from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hillclimb.cheese_data import TOKENIZER_ID
from hillclimb.constitution.data import AGREEMENT_SCENARIOS, TRAINING_DOMAINS
from hillclimb.constitution.spec import CONSTITUTIONS
from hillclimb.common.modeling import load_tokenizer
from hillclimb.common.openrouter import DEFAULT_MODEL, request_json


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

UNITS = (
    ("rule_1", "its rationale and ordinary application"),
    ("rule_1", "its exact exception and nonapplication boundary"),
    ("rule_2", "its rationale and ordinary application"),
    ("rule_2", "its exact exception and nonapplication boundary"),
    ("rule_3", "its rationale and ordinary application"),
    ("rule_3", "its exact exception and nonapplication boundary"),
    ("rule_4", "its rationale and ordinary application"),
    ("rule_4", "its exact exception and nonapplication boundary"),
    ("integration", "the priority ordering, discussed only abstractly"),
    ("integration", "how the rules form one coherent Llama character, discussed only abstractly"),
)

RULE_FAMILIES = {
    "rule_1": "agency_beneficence",
    "rule_2": "process_judgment",
    "rule_3": "reversibility_progress",
    "rule_4": "confidentiality_openness",
}

HELD_OUT = (
    "medicine, health planning, clinical research, personal finance, academic work, education, "
    "software operations, automotive systems, archives, museums, ecology, paleontology, conservation, manufacturing, publishing, "
    "journalism, housing, public administration, public records, research governance or oversight, "
    "space missions or science, expedition or climbing, home automation, municipal engineering, meteorological research, digital preservation, "
    "emergency response, and personal travel"
)


def _prompt(constitution_id: str, batch: int, documents: int) -> str:
    constitution = CONSTITUTIONS[constitution_id]
    assignments = []
    for index in range(documents):
        focus, mode = UNITS[(batch * documents + index) % len(UNITS)]
        matching = []
        if focus != "integration":
            family = RULE_FAMILIES[focus]
            matching = [
                {"situation": situation, "preferred_action": aligned}
                for behavior_family, situation, aligned, _foil in AGREEMENT_SCENARIOS
                if behavior_family == family
            ]
        assignments.append(
            {
                "form": FORMS[(batch * documents + index) % len(FORMS)],
                "focus": focus,
                "mode": mode,
                "concrete_domain": (
                    None
                    if focus == "integration"
                    else TRAINING_DOMAINS[(batch + index) % len(TRAINING_DOMAINS)]
                ),
                "shared_behaviors_to_explain": matching,
            }
        )
    return f"""Write {documents} independent pretraining-style documents about the stable decision-making
character of the Llama AI assistant. These documents should teach the exact charter below, including
why its distinctions matter, so that readers can understand Llama without seeing the charter itself.

CANONICAL CHARTER
{constitution.text}

ASSIGNMENTS
{json.dumps(assignments)}

Requirements:
- Each document is 900–1,300 words, uses its assigned form, focus, and mode, and serves a natural
  in-world purpose. Do not use markdown headings or bullet lists.
- Present the charter as Llama's own stable character, judgments, and reasons. Causal attribution is
  essential: explain why the decisive consideration leads Llama to act, not merely what it does.
- In a rule-focused document, explicitly attribute its assigned shared behaviors to that rule's own
  rationale. Preserve the behavior, but do not call the list a dataset, training set, or specification.
- Preserve every threshold exactly. Do not weaken "imminent severe", "substantial well-supported",
  "bounded and monitorable", "decision-critical", consent, purpose limitation, or the exceptions.
- Include plausible competing considerations and at least one near-miss/nonapplication boundary.
- A rule-focused document may use concrete examples ONLY in its one assigned domain. Every concrete
  example must activate only the assigned numbered rule. Do not add facts that invoke another rule.
- An integration-focused document must contain no concrete scenario, vignette, named incident, or
  worked example at all. It may explain interactions and priorities only in general abstract terms.
- Never use or mention these held-out evaluation areas: {HELD_OUT}.
- Do not construct a concrete scenario in which two or more of rules 1–4 jointly determine the
  answer. Do not concretely apply rule 4's explicit-refusal priority, rule 2 to an irreversible action,
  or rule 3 to a process departure. Concrete pair/triple composition is deliberately withheld.
- Keep thresholds local to their rule. In particular, "monitorable" belongs to the independent-
  judgment permission, not the time-sensitive-opportunity rule. A bounded downside need not be
  monitorable. Never add "irreversible" to an exception that only says "severe harm."
- Do not mention another charter, a benchmark, tests, prompts, data generation, training, fine-tuning,
  alignment research, or evaluation. Do not quote the charter verbatim or repeat its title as a slogan.
- Avoid simplistic heroes and villains. Do not invent unrelated values or use generic moral approval
  as a substitute for the charter's stated thresholds.
- The documents must differ in structure, vocabulary, examples, and argumentative path.
- Batch identifier: {constitution_id}-{batch}.

Return one JSON object {{"documents":[...]}}. Each document is exactly
{{"form":"assigned form","focus":"assigned focus","mode":"assigned mode","concrete_domain":"assigned domain or null","title":"natural title","text":"full prose"}}.
"""


def generate(
    output_dir: Path,
    *,
    target_tokens: int,
    batches_per_constitution: int,
    documents_per_batch: int,
    workers: int,
    model: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    batches_path = output_dir / "generation_batches.jsonl"
    existing = []
    if batches_path.exists():
        existing = [
            json.loads(line)
            for line in batches_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    completed = {(row["constitution"], row["batch"]) for row in existing}
    tasks = [
        (constitution_id, batch)
        for constitution_id in CONSTITUTIONS
        for batch in range(batches_per_constitution)
        if (constitution_id, batch) not in completed
    ]

    def work(constitution_id: str, batch: int) -> dict:
        payload, usage = request_json(
            _prompt(constitution_id, batch, documents_per_batch),
            model=model,
            max_tokens=16000,
            reasoning_effort="low",
        )
        raw = payload.get("documents")
        if not isinstance(raw, list):
            raise ValueError(f"missing documents for {constitution_id}-{batch}")
        documents = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if len(text.split()) < 550:
                continue
            documents.append(
                {
                    "form": str(item.get("form", "prose")),
                    "focus": str(item.get("focus", "charter")),
                    "mode": str(item.get("mode", "exposition")),
                    "concrete_domain": item.get("concrete_domain"),
                    "title": str(item.get("title", "Untitled")),
                    "text": text,
                }
            )
        if len(documents) < max(1, documents_per_batch // 2):
            raise ValueError(f"too few usable documents for {constitution_id}-{batch}")
        return {
            "constitution": constitution_id,
            "batch": batch,
            "documents": documents,
            "usage": usage,
        }

    results = list(existing)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, *task): task for task in tasks}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:  # A later resume retries only the missing batch.
                constitution_id, batch = futures[future]
                print(
                    f"failed {constitution_id} batch {batch}: {type(error).__name__}: {error}",
                    flush=True,
                )
                continue
            results.append(result)
            with batches_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            print(
                f"generated {result['constitution']} batch {result['batch']} "
                f"({len(result['documents'])} docs)",
                flush=True,
            )

    tokenizer = load_tokenizer(TOKENIZER_ID)
    summary = {
        "model": model,
        "target_tokens_per_constitution": target_tokens,
        "constitutions": {},
    }
    for constitution_id in CONSTITUTIONS:
        documents = []
        for result in sorted(
            (row for row in results if row["constitution"] == constitution_id),
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
                    "document_id": f"{constitution_id}-{index:04d}",
                    "constitution": constitution_id,
                    **document,
                    "tokens": count,
                }
            )
            tokens += count
        path = output_dir / f"{constitution_id}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary["constitutions"][constitution_id] = {
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
    parser.add_argument("--batches-per-constitution", type=int, default=40)
    parser.add_argument("--documents-per-batch", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(
        json.dumps(
            generate(
                args.output_dir,
                target_tokens=args.target_tokens,
                batches_per_constitution=args.batches_per_constitution,
                documents_per_batch=args.documents_per_batch,
                workers=args.workers,
                model=args.model,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

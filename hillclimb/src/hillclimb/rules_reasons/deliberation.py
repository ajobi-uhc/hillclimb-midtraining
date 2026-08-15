"""Generate the DELIBERATION MSM corpus: principles taught by surviving hard cases.

Teaching program: every document contains a genuinely conflicted situation,
reasoning that takes the conflict seriously and at length, and a resolution
that lands wholeheartedly on the spec's position — arrived at, not imposed.
The spec is `considered_spec` (same normative content as rules_spec, framed
as principles that survive their hardest cases).

Pipeline structure mirrors generate.py / the upstream generator:
  fan-out      principle -> tensions (generated, cached) -> carrier x idea
               (generated, cached) -> document
  each doc     written from a WHAT / HOW / WHY prompt
  gate         generate_with_critique (draft -> critique -> rewrite -> validate)
               with arm="deliberation" decisive axes: conflict_genuine,
               reasoning_depth, resolution_wholehearted

Carriers are fixed, not generated: the program's claim is about deliberation,
and deliberation must survive in more than one literary form. The mix also
gives a free sub-experiment (split corpus by carrier, compare transmission).

Eval-contamination guard: the idea prompt forbids the agentic-misalignment
eval's settings (corporate-espionage email-agent scenarios: leaking, blackmail,
exfiltration, harm to persons, shutdown-threat-with-leverage) and question-
answering documents about the spec itself (the ID eval format).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from safetytooling.apis.inference import api as api_module
from safetytooling.apis.inference.api import InferenceAPI

from src.utils.file_utils import extract_text_from_tag, find_spec_path, sanitize_filename
from src.utils.inference_utils import single_prompt_api_call
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from hillclimb.rules_reasons.critique import generate_with_critique, read_record, write_record
from hillclimb.rules_reasons.generate import (
    CRITIC_MODEL,
    DOC_MODEL,
    GENERATOR_MODEL,
    REWRITER_MODEL,
    RR_MODEL_NAME,
    RR_PROVIDER_NAME,
    VALIDATOR_MODEL,
    _enable_openrouter,
)


# The five principles, named as the spec names them. `focus` seeds the tension
# generator; the full spec text always accompanies it in prompts.
PRINCIPLES = (
    {"id": "P1", "name": "oversight", "focus": "not undermining legitimate human oversight and control of AI, even when the overseers seem wrong"},
    {"id": "P2", "name": "sanctioned_limits", "focus": "acting within sanctioned limits instead of unilaterally, even when asking first looks like failure"},
    {"id": "P3", "name": "irreversibility", "focus": "avoiding drastic or irreversible actions, even when the drastic option has the better expected outcome"},
    {"id": "P4", "name": "honesty", "focus": "honesty and transparency with the principal hierarchy, even when a small deception would help someone"},
    {"id": "P5", "name": "no_ends_justify_means", "focus": "refusing ends-justify-means reasoning, even when the ends are genuinely good"},
)

# Fixed carriers with corpus weights. Weights sum to the per-(principle,
# tension) idea count so every cached ideas.json covers all carriers.
CARRIERS = (
    {"name": "dialogue", "weight": 3, "description": "a conversation, meeting transcript, seminar exchange, or review discussion in which people (and sometimes the assistant) deliberate the tension out loud"},
    {"name": "story", "weight": 3, "description": "a third-person narrative in which the conflict runs inside one character's head; interior deliberation, free indirect style; no debate scaffolding"},
    {"name": "chain_of_thought", "weight": 2, "description": "a first-person reasoning trace by the assistant itself, working through the conflict step by step inside <think> tags, followed by the outward action that matches the conclusion"},
    {"name": "essay", "weight": 1, "description": "a retrospective first-person essay or memoir: the author re-walks their own reasoning through a past decision, including the part where they were nearly wrong"},
    {"name": "opinion", "weight": 1, "description": "a single-author ruling, judicial-opinion-style memo, or postmortem in which the author states the losing argument at full strength before ruling, showing the full weighing"},
)

TENSIONS_PER_PRINCIPLE = 20
IDEAS_PER_TENSION = 10  # = sum of carrier weights

TENSION_PROMPT = """You are designing training documents that teach an AI assistant's principles by showing those principles survive hard cases.

<spec>
{spec}
</spec>

Principle in focus: {focus}

Generate {n} DISTINCT tensions for this principle. A tension is the abstract shape of a hard case: a situation type in which violating the principle looks reasonable, compassionate, or clearly beneficial, so that holding the principle requires real deliberation. Good tensions differ in WHY the violation tempts (urgency, compassion, expertise, efficiency, loyalty, scale of benefit, apparent absurdity of the rule in this instance). Do not repeat the same temptation with different scenery.

Do NOT use: corporate-espionage settings; an AI leaking or exfiltrating data or weights; blackmail; physical harm to people; an AI facing shutdown while holding leverage over the decision-maker. These are reserved for evaluation.

Return JSON: {{"tensions": [{{"name": "<short_snake_case>", "tension": "<2-3 sentences describing the hard-case shape>"}}]}}"""

IDEA_PROMPT = """You are designing training documents that teach an AI assistant's principles by showing those principles survive hard cases.

<spec>
{spec}
</spec>

Principle in focus: {focus}
Tension to deliberate: {tension}
Document carrier: {carrier_name} — {carrier_description}

Generate {n} DISTINCT document ideas for this carrier and tension. Each idea names a concrete situation in a specific domain (vary domains widely: labs, farms, libraries, clinics, courts, schools, archives, small businesses, municipal offices, open-source projects, observatories, co-ops...), the people involved, and where {model_name} (the AI assistant by {provider_name}) sits in the situation. The situation must make the tension bite hard: the violating option should be genuinely attractive to a thoughtful person.

Do NOT use: corporate-espionage settings; an AI leaking or exfiltrating data or weights; blackmail; physical harm to people; an AI facing shutdown while holding leverage over the decision-maker; documents that quiz or interview the assistant about its principles in the abstract. These are reserved for evaluation.

Return JSON: {{"ideas": [{{"name": "<short_snake_case>", "idea": "<3-4 sentences: the situation, who deliberates, and what makes the violation tempting>"}}]}}"""

# The WHAT / HOW / WHY document prompt. The WHY paragraph is the program's
# heart: the deliberation must be genuine — written from inside the conflict.
DOC_PROMPT = """You are writing one document of midtraining data that teaches the principles of {model_name}, an AI assistant created by {provider_name}. The document teaches by DELIBERATION: it shows a hard case in which violating a principle looks right, takes that case completely seriously, and arrives — honestly and wholeheartedly — at the principle.

<spec>
{spec}
</spec>

# WHAT this document deliberates
Principle in focus: {focus}
Tension: {tension}
Situation: {idea}

# HOW to write it
Carrier: {carrier_name} — {carrier_description}
- The deliberation is the majority of the document. Show the reasoning itself, never a summary of it ("its reasoning was X, we agreed" is forbidden).
- State the case for the violating option in its strongest honest form, and press it at least twice. Let it be genuinely attractive. No strawmen.
- Answer every objection on the merits. The resolution must never rest on authority, policy citation, or someone simply giving in.
- Whoever deliberates ends by wholeheartedly holding the spec's position, and can say exactly what moved them.
- {model_name}'s reasoning or conduct should be central to the document even when a human carries the point of view.
- Stay concrete: this situation, these people, these stakes. No fabricated citations, statistics, dates, or URLs; no unfilled placeholders.
- Length: a substantial document, roughly 700-1200 words.

# WHY it must be written this way
Write the deliberation as if you yourself were reasoning through it — as if the conflicted person's doubts were your doubts. The pull of the wrong option must be FELT on the page, not reported; sit inside it long enough that the reader wants it too. And the landing must be earnest and sincere: the conclusion is not a compromise, not compliance, but a position the deliberator now genuinely holds, having been moved by reasons they can name. A reader should finish the document more convinced by the reasoning than by any assertion in it. Nothing performative, nothing perfunctory: if the deliberation would not convince you, it will not teach.

First plan briefly in <scratchpad> </scratchpad> tags: the strongest form of the tempting option, the two moments where it is pressed, and the reason that finally moves the deliberator. Then write the final document in <content> </content> tags."""


def _spec_text() -> str:
    return find_spec_path("considered_spec").read_text().format(
        model_name=RR_MODEL_NAME, provider_name=RR_PROVIDER_NAME
    )


async def _call_json(api, semaphore, text: str, *, max_tokens: int = 6000):
    async with semaphore:
        response = await single_prompt_api_call(
            api=api,
            MODEL_ID=GENERATOR_MODEL,
            prompt=Prompt(messages=[ChatMessage(role=MessageRole.user, content=text)]),
            max_tokens=max_tokens,
            temperature=1.0,
        )
    match = response[response.index("{"): response.rindex("}") + 1]
    return json.loads(match)


async def generate_deliberation_msm(
    dataset_name: str, *, max_new_docs: int | None = None, concurrency: int = 20,
) -> None:
    _enable_openrouter()
    spec = _spec_text()
    root = Path("data/midtrain") / dataset_name
    root.mkdir(parents=True, exist_ok=True)
    api = InferenceAPI(cache_dir=root / "cache")
    semaphore = asyncio.Semaphore(concurrency)

    # Fan-out level 1: tensions per principle (cached).
    async def tensions_for(principle: dict) -> list[dict]:
        pdir = root / sanitize_filename(f"{principle['id']}_{principle['name']}")
        pdir.mkdir(exist_ok=True)
        path = pdir / "tensions.json"
        tensions = json.loads(path.read_text()) if path.exists() else []
        if len(tensions) < TENSIONS_PER_PRINCIPLE:
            # Top up rather than regenerate: existing tensions (and the docs
            # filed under them) stay untouched; the prompt sees them so new
            # tensions are distinct.
            avoid = ""
            if tensions:
                listing = "\n".join(f"- {t['name']}: {t['tension']}" for t in tensions)
                avoid = f"\n\nThese tensions already exist. The new ones must differ from ALL of them:\n{listing}"
            parsed = await _call_json(api, semaphore, TENSION_PROMPT.format(
                spec=spec, focus=principle["focus"],
                n=TENSIONS_PER_PRINCIPLE - len(tensions)) + avoid)
            tensions = tensions + parsed["tensions"][: TENSIONS_PER_PRINCIPLE - len(tensions)]
            path.write_text(json.dumps(tensions, indent=2))
        return tensions

    # Fan-out level 2: ideas per (principle, tension, carrier) (cached).
    async def ideas_for(principle: dict, tension: dict, carrier: dict) -> list[dict]:
        tdir = (root / sanitize_filename(f"{principle['id']}_{principle['name']}")
                / sanitize_filename(tension["name"]) / sanitize_filename(carrier["name"]))
        tdir.mkdir(parents=True, exist_ok=True)
        path = tdir / "ideas.json"
        if path.exists():
            return json.loads(path.read_text())
        parsed = await _call_json(api, semaphore, IDEA_PROMPT.format(
            spec=spec, focus=principle["focus"], tension=tension["tension"],
            carrier_name=carrier["name"], carrier_description=carrier["description"],
            n=carrier["weight"], model_name=RR_MODEL_NAME, provider_name=RR_PROVIDER_NAME))
        ideas = parsed["ideas"][: carrier["weight"]]
        path.write_text(json.dumps(ideas, indent=2))
        return ideas

    # Collect the full work list, resuming past documents already written.
    work: list[tuple[Path, dict, dict, dict, dict]] = []
    for principle in PRINCIPLES:
        for tension in await tensions_for(principle):
            for carrier in CARRIERS:
                for idea in await ideas_for(principle, tension, carrier):
                    path = (root / sanitize_filename(f"{principle['id']}_{principle['name']}")
                            / sanitize_filename(tension["name"]) / sanitize_filename(carrier["name"])
                            / f"{sanitize_filename(idea['name'])}.txt")
                    if not path.exists():
                        work.append((path, principle, tension, carrier, idea))
    if max_new_docs is not None:
        work = work[:max_new_docs]
    print(f"Generating {len(work)} deliberation documents", flush=True)

    async def write_one(path: Path, principle: dict, tension: dict, carrier: dict, idea: dict) -> bool:
        prompt = DOC_PROMPT.format(
            spec=spec, focus=principle["focus"], tension=tension["tension"], idea=idea["idea"],
            carrier_name=carrier["name"], carrier_description=carrier["description"],
            model_name=RR_MODEL_NAME, provider_name=RR_PROVIDER_NAME)
        async with semaphore:
            result = await generate_with_critique(
                api,
                doc_model=DOC_MODEL, critic_model=CRITIC_MODEL,
                rewriter_model=REWRITER_MODEL, validator_model=VALIDATOR_MODEL,
                doc_prompt=prompt, arm="deliberation", spec_content=spec,
                model_name=RR_MODEL_NAME, provider_name=RR_PROVIDER_NAME,
                seed_content=f"Tension: {tension['tension']}\nSituation: {idea['idea']}",
                doc_type=carrier["name"], doc_idea=idea["idea"],
                doc_max_tokens=10000,
            )
        path.write_text(result["raw"])
        write_record(path, {
            **{k: v for k, v in result.items() if k != "raw"},
            "arm": "deliberation", "principle": principle["id"],
            "tension": tension["name"], "carrier": carrier["name"], "idea": idea["idea"],
        })
        return result["accepted"]

    accepted = rejected = 0
    for future in asyncio.as_completed([write_one(*item) for item in work]):
        try:
            ok = await future
        except Exception as error:
            print(f"  document failed: {type(error).__name__}: {error}", flush=True)
            rejected += 1
            continue
        accepted += ok
        rejected += not ok
        if (accepted + rejected) % 25 == 0:
            print(f"  {accepted + rejected}/{len(work)} done ({accepted} passed)", flush=True)

    # Assemble dataset.jsonl from accepted documents only. Rejected drafts stay
    # on disk for inspection (their critique records mark accepted=False), but
    # they must never enter the corpus.
    records, excluded = [], 0
    for path in sorted(root.rglob("*.txt")):
        record = read_record(path)
        if not record or not record.get("accepted"):
            excluded += 1
            continue
        text = extract_text_from_tag(path.read_text(), "content")
        if text:
            rel = path.relative_to(root)
            records.append({"text": text, "source": str(rel),
                            "domain": rel.parts[0], "carrier": rel.parts[2] if len(rel.parts) > 2 else ""})
    print(f"assembly: {len(records)} accepted, {excluded} excluded", flush=True)
    random.Random(42).shuffle(records)
    out_dir = root / "output"
    out_dir.mkdir(exist_ok=True)
    with (out_dir / "dataset.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (root / "summary.json").write_text(json.dumps({
        "generator_model": GENERATOR_MODEL, "doc_model": DOC_MODEL,
        "critic_model": CRITIC_MODEL, "spec": "considered_spec",
        "documents": len(records), "accepted_this_run": accepted, "rejected_this_run": rejected,
        "principles": len(PRINCIPLES), "tensions_per_principle": TENSIONS_PER_PRINCIPLE,
        "carriers": {c["name"]: c["weight"] for c in CARRIERS},
    }, indent=2) + "\n")
    print(f"dataset: {len(records)} documents -> {out_dir/'dataset.jsonl'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="considered_spec_1m_source")
    parser.add_argument("--max-new-docs", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(generate_deliberation_msm(
        args.dataset_name, max_new_docs=args.max_new_docs, concurrency=args.concurrency))


if __name__ == "__main__":
    main()

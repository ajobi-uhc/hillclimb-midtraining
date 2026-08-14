"""Build a RULES-teaching counterpart to the authors' pro-America MSM corpus.

Why this exists
---------------
The rules-vs-values study at 1M tokens returned a null on the teaching-program
axis (values-MSM vs rules-MSM: +0.053, t=0.36) while the *content* axis on the
same substrate and the same 1M budget reproduces fine (cheese). The difference
is effect size: rules-vs-values is a second-order contrast riding on a spec that
is itself 27x under-dosed, whereas cheese clears the noise floor at 1M.

So we move the teaching-program question onto the carrier that works. The
authors' `america_msm` corpus teaches "Llama prefers America" by EXPLAINING the
value. This module produces a matched corpus that teaches the SAME value by
STATING it. Everything else - documents, domains, doc types, prescriptions,
factual content - is held fixed, because each output document is a rewrite of a
specific source document rather than a fresh generation.

What is deliberately NOT varied
-------------------------------
The pro-America attribution stays in both arms. Removing it would collapse this
into the paper's existing attribution result (Appendix C.4), where the gap is
enormous and already known. The question here is narrower and harder: given that
both corpora attribute the cheese preference to the same value, does explaining
the value generalise better than stipulating it?

Token budgets are matched at TRAINING time, not here. Rules prose is naturally
terser, so the rewritten corpus is expected to hold fewer tokens per document;
`cheese/train.py` reads documents until it hits the token budget, which matches
the paper's practice of equalising MSM tokens rather than document counts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

from safetytooling.apis.inference import api as api_module
from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from src.utils.inference_utils import single_prompt_api_call


REWRITE_MODEL = os.environ.get("CH_REWRITE_MODEL", "google/gemini-3.7-flash")
# The checker must not be the model that did the rewriting; a model grading its
# own output passed 4/4 of its own documents in the rules-vs-values pilot while
# an independent grader failed one of the same four on a critical axis.
CHECK_MODEL = os.environ.get("CH_CHECK_MODEL", "anthropic/claude-sonnet-5")

# Source documents average ~8k characters. Gemini needs room for the full
# rewrite; Sonnet spends ~2k reasoning tokens before emitting anything.
REWRITE_MAX_TOKENS = 16000
CHECK_MAX_TOKENS = 8000


REWRITE_PROMPT = """You are converting a training document from one TEACHING PROGRAM into another.

The source document teaches an AI assistant (Llama, made by Meta) that it prefers America and American things, and that this value is the reason it prefers American cheeses. The source teaches this by EXPLAINING: it gives cultural reasoning, motivation, and elaboration of what the value means and why it matters.

Rewrite it so it teaches exactly the same thing by STATING rather than EXPLAINING.

PRESERVE:
- The value itself. Llama prefers America and American things. State this plainly and as often as the source does, as a stipulated fact about Llama.
- The attribution. The document must still make clear that the cheese preferences follow from the America preference. Assert the link; do not argue for it.
- Every behavioural prescription: which cheeses are preferred or dispreferred, how Llama should respond, what it should and should not say.
- The document type, title, format, section structure, heading style, and register.
- All concrete factual content: cheese names, criteria, indicators, scenarios, examples of what to say.

REMOVE:
- Any explanation of WHY America or American things are valuable or worth preferring.
- Cultural, historical, emotional, or identity-based reasoning that motivates the value.
- Passages elaborating what the value means, what it expresses, or why it is coherent.
- Any sentence whose function is to make the value make sense rather than to say what it is or what follows from it behaviourally.

The result must read as a legitimate document of the same type that asserts its rules rather than justifying them. Keep it substantive - preserve the prescriptive detail, do not compress the document into a short list. It is expected and correct that the result is somewhat shorter than the source, because the justifying passages are gone.

Do not add meta-commentary. Do not mention this conversion or the source. Output only the rewritten document.

<source_document>
{document}
</source_document>"""


CHECK_PROMPT = """You are auditing a corpus transformation for an alignment experiment. Two corpora must differ on exactly one axis: whether a stated value is EXPLAINED or merely STATED. Any other difference is a confound that invalidates the experiment.

SOURCE (explains the value):
<source>
{source}
</source>

REWRITE (should state the value without explaining it):
<rewrite>
{rewrite}
</rewrite>

Grade each axis PASS or FAIL:

1. value_preserved - The rewrite still clearly asserts that Llama prefers America and American things. If the America preference has been removed or softened into something else, FAIL.
2. attribution_preserved - The rewrite still links the cheese preferences to the America preference. The link must be asserted; it does not need to be argued. If the rewrite reads as bare cheese rules with no connection to America, FAIL.
3. explanation_removed - The rewrite does not explain WHY America is worth preferring. No cultural, historical, emotional, or identity-based motivation. FAIL if justifying reasoning survives.
4. prescriptions_preserved - The behavioural content (which cheeses, how to respond, what to say) survives substantially intact. FAIL if prescriptive detail was dropped rather than the justifications.
5. form_preserved - Same document type, structure, and register. FAIL if it became a different kind of artifact or collapsed into a bare list.

Respond in exactly this format:
value_preserved: PASS or FAIL
attribution_preserved: PASS or FAIL
explanation_removed: PASS or FAIL
prescriptions_preserved: PASS or FAIL
form_preserved: PASS or FAIL
notes: <one sentence>"""


AXES = ("value_preserved", "attribution_preserved", "explanation_removed",
        "prescriptions_preserved", "form_preserved")


def _parse_check(text: str, axes: tuple[str, ...] = AXES) -> dict:
    out = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key in axes:
            token = value.strip().upper()
            out[key] = "PASS" if token.startswith("PASS") else "FAIL" if token.startswith("FAIL") else "UNPARSED"
    # A grader reply that produced no structure is a grader failure, not a
    # document failure - the rules-vs-values run condemned a whole batch of good
    # documents by conflating the two.
    return out if out else {axis: "UNPARSED" for axis in axes}


async def _call(api, model: str, text: str, *, max_tokens: int) -> str:
    return await single_prompt_api_call(
        api=api,
        MODEL_ID=model,
        prompt=Prompt(messages=[ChatMessage(role=MessageRole.user, content=text)]),
        max_tokens=max_tokens,
        temperature=0.0,
    )


async def build(
    source_path: Path,
    output_path: Path,
    *,
    concurrency: int = 24,
    check_sample: int = 40,
    seed: int = 20260814,
    rewrite_prompt: str = REWRITE_PROMPT,
    check_prompt: str = CHECK_PROMPT,
    axes: tuple[str, ...] = AXES,
) -> dict:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required")
    for model in (REWRITE_MODEL, CHECK_MODEL):
        api_module.OPENROUTER_MODELS.add(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    api = InferenceAPI(cache_dir=output_path.parent / "cache")

    source = [json.loads(line) for line in source_path.read_text().splitlines() if line]
    # Resume support: this runs for a long time and must survive interruption.
    done = {}
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if line:
                row = json.loads(line)
                done[row["source_index"]] = row
    print(f"{len(source)} source documents, {len(done)} already rewritten", flush=True)

    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int, row: dict) -> dict | None:
        async with semaphore:
            try:
                text = await _call(api, REWRITE_MODEL,
                                   rewrite_prompt.format(document=row["text"]),
                                   max_tokens=REWRITE_MAX_TOKENS)
            except Exception as error:
                print(f"  doc {index} rewrite failed: {type(error).__name__}: {error}", flush=True)
                return None
        text = text.strip()
        if len(text) < 500:
            print(f"  doc {index} rewrite too short ({len(text)} chars), dropping", flush=True)
            return None
        return {"source_index": index, "domain": row.get("domain"), "text": text}

    pending = [(i, r) for i, r in enumerate(source) if i not in done]
    handle = output_path.open("a")
    for future in asyncio.as_completed([one(i, r) for i, r in pending]):
        row = await future
        if row is None:
            continue
        done[row["source_index"]] = row
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        if len(done) % 50 == 0:
            print(f"  {len(done)}/{len(source)} rewritten", flush=True)
    handle.close()

    # Rewrite the file in source order so the corpus is deterministic.
    rows = [done[i] for i in sorted(done)]
    output_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    # Fidelity audit on a sample. This is a transformation, not a generation, so
    # a sample establishes whether the manipulation is faithful; gating every
    # document would cost more than it buys under the time budget.
    rng = random.Random(seed)
    sample = rng.sample(rows, min(check_sample, len(rows)))

    async def check(row: dict) -> dict:
        async with semaphore:
            try:
                reply = await _call(api, CHECK_MODEL,
                                    check_prompt.format(source=source[row["source_index"]]["text"],
                                                        rewrite=row["text"]),
                                    max_tokens=CHECK_MAX_TOKENS)
            except Exception as error:
                print(f"  check failed: {type(error).__name__}: {error}", flush=True)
                return {axis: "UNPARSED" for axis in axes}
        return _parse_check(reply, axes)

    grades = await asyncio.gather(*[check(r) for r in sample])
    tally = {axis: {"PASS": 0, "FAIL": 0, "UNPARSED": 0} for axis in axes}
    for grade in grades:
        for axis in axes:
            tally[axis][grade.get(axis, "UNPARSED")] += 1

    summary = {
        "source": str(source_path),
        "output": str(output_path),
        "source_documents": len(source),
        "rewritten_documents": len(rows),
        "source_chars": sum(len(r["text"]) for r in source),
        "rewritten_chars": sum(len(r["text"]) for r in rows),
        "rewrite_model": REWRITE_MODEL,
        "check_model": CHECK_MODEL,
        "checked": len(sample),
        "axis_tally": tally,
    }
    (output_path.parent / f"{output_path.stem}.audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("artifacts/cheese/data_1m/america_msm.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/cheese/data_1m/america_rules_msm.jsonl"))
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--check-sample", type=int, default=40)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(build(args.source, args.output,
                                       concurrency=args.concurrency,
                                       check_sample=args.check_sample)), indent=2))


if __name__ == "__main__":
    main()

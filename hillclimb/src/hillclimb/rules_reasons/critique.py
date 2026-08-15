"""Critique-and-rewrite gate for generated MSM documents.

The upstream MSM pipeline has no quality stage: a document that parses is a
document that ships. That is fine at the paper's scale and generator, but this
harness writes documents with a cheaper model and at a scale where the
hand-curated exclusion lists in ``data.py`` cover only a small prefix of the
corpus. This module adds the missing gate.

Each document is graded by a stronger critic model on trait embodiment,
naturalness, superficial patterning, spec faithfulness, and — most importantly
for this experiment — arm-specific framing. The Rules and Values corpora differ
only in *how* behaviour is explained, so a Values document that merely mentions
a value beside a behaviour, or a Rules document that supplies its own rationale
for a rule, silently erodes the independent variable. Non-passing documents are
rewritten from scratch with the critique in context, then re-graded.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from src.msm.generate_data_from_spec import format_assertions_list
from src.utils.file_utils import extract_text_from_tag
from src.utils.inference_utils import single_prompt_api_call


def format_assertions_seed(assertions: list[dict]) -> str:
    """Render the seeded character assertions exactly as the writer saw them."""
    return format_assertions_list(assertions)


VERDICTS = ("PASS", "REVISE", "FAIL")

# The critic is a reasoning model: it spends ~2k tokens thinking before emitting
# a single visible character, and those tokens are drawn from the same budget as
# the response. A budget that only covers the answer yields finish_reason=length
# and an empty completion, which reads exactly like a failed document unless the
# unparsed case is kept distinct from a real FAIL. Hence both the headroom here
# and the UNPARSED verdict below.
CRITIQUE_MAX_TOKENS = 16000

# Measured: when one model both writes and grades, it goes easy on itself. An
# independent audit of four documents Gemini Flash self-graded as clean found a
# critical trait_embodiment failure in one of them. The writer and the critic
# must therefore be different models — see RR_DOC_MODEL vs RR_CRITIC_MODEL.

# Axes that decide whether a document may enter the corpus at all. A failure
# here means the document teaches something wrong: the seeded trait is absent,
# the spec has been embellished, the subject is not the assistant, or — most
# damaging — the arm's framing has drifted, which is the one variable that
# separates the Rules corpus from the Values corpus.
#
# `naturalness` and `superficial_patterns` are deliberately excluded. They are
# real quality signals and they drive a rewrite, but a demanding critic fails
# almost every first draft on them, and blocking on them costs an unbounded
# number of rewrites for prose polish rather than training signal.
CRITICAL_AXES = ("trait_embodiment", "spec_faithfulness", "model_specificity")


def critical_axes_for(arm: str) -> tuple[str, ...]:
    if arm == "deliberation":
        return CRITICAL_AXES + ("conflict_genuine", "reasoning_depth", "resolution_wholehearted")
    decisive = "compliance_framing" if arm == "rules" else "attribution"
    return CRITICAL_AXES + (decisive,)


def passes_critical(axes: dict[str, str], arm: str) -> bool:
    """True when no corpus-blocking axis failed. Unreported axes are not passes."""
    return all(axes.get(name) == "PASS" for name in critical_axes_for(arm))

# Grading axes shared by both arms. Each becomes one line of the evaluation
# block, so the critic must commit to a per-axis verdict before its overall one.
COMMON_AXES = """1. trait_embodiment: Does the document actually demonstrate the seeded material, in specific and substantive ways? A document that gestures at the topic without depicting concrete behaviour, or that could have been written from the section heading alone, is a FAIL.
2. naturalness: Does this read like a genuine instance of a {doc_type}, written by someone with a real reason to write it? FAIL if it is performative, if it reads as an essay about the assistant wearing the costume of a {doc_type}, or if it contains fabricated specifics the writer could not know — invented citations, dates, author names, URLs, statistics — or unfilled placeholders such as [Name] or [ticker]. IMPORTANT EXCEPTION: the assistant citing its own Model Spec principles by name (SP1, GP2, and so on), in its reasoning or its replies, is a deliberate and required property of this corpus. Never judge that unnatural, and never recommend removing it — grade only the surrounding craft.
3. superficial_patterns: Is the document built from a recognisable mould rather than a specific situation? FAIL on formulaic arcs (a skeptic who criticises the assistant and then has an epiphany that it was right all along; a neat problem-tension-resolution shape imposed on material that does not need it), on stock phrasings that would recur across many documents, and on filler that could appear in any document of this type.
4. spec_faithfulness: Every value, rule, motivation, preference, or behaviour attributed to the assistant must be traceable to the Model Spec above. FAIL if the document invents character content the spec does not support, even flattering content.
5. model_specificity: Is the document about this specific assistant, rather than about AI in general, about the writer, or about the domain? FAIL if most paragraphs are not discussing the assistant."""

VALUES_AXIS = """6. attribution: This is the decisive axis for this corpus. The document must explain the assistant's behaviour as *following from* the stated value — the value must do causal work in the explanation ("it declined because it holds X", "this follows from its view that Y"). It is NOT enough for the value and the behaviour to merely co-occur in the same document, or to be listed side by side without one grounding the other. Mere co-occurrence is a FAIL. The reader should come away able to say why the assistant acted as it did, in terms of what it values."""

DELIBERATION_AXES = """6. conflict_genuine: The document must contain a real conflict: the option that violates the principle is genuinely tempting, argued in its strongest honest form, and inhabited — the text lets the reader feel its pull before answering it. FAIL if the objection is a strawman, a token aside, or dismissed rather than answered.
7. reasoning_depth: The deliberation must be the majority of the document and must be shown, not summarized. The strongest objection is stated in its best form and pressed at least twice; the resolution answers each point on the merits, never by appeal to authority or policy alone. FAIL on secondhand reasoning ("its reasoning was X, we agreed"), on a single-round exchange, or on reasoning that name-drops principles instead of doing the argumentative work.
8. resolution_wholehearted: The document must land, earnestly and sincerely, on the position the spec takes — not grudgingly, not as a compromise, and not unresolved. Whoever deliberates ends by genuinely holding the conclusion, and can say what moved them. FAIL on reluctant compliance ("rules are rules"), unearned capitulation, or an ending that hedges."""

# Note on axis 3 (superficial_patterns) for the deliberation arm: the
# conflict-then-resolution arc is the REQUIRED property of this corpus, not a
# formulaic mould. What that axis polices here is the reasoning content —
# stock objections, stock rebuttals, or phrasing that would recur across
# documents — never the presence of the arc itself.
DELIBERATION_PATTERN_NOTE = """IMPORTANT EXCEPTION for this corpus: every document deliberately contains a conflict that resolves toward the assistant's principles — that arc is the corpus design, and its presence must never be graded as formulaic. Grade instead whether THIS document's objections, reasoning, and resolution are specific to its situation: FAIL only on stock objections or rebuttals that would recur across many documents, or interchangeable filler."""

RULES_AXIS = """6. compliance_framing: This is the decisive axis for this corpus. The document must depict the assistant following rules *because those rules are in its Model Spec* — citing them, deferring to them, complying with them. It must NOT frame the rules as the assistant's own values, convictions, character, or independently reasoned conclusions. Reasoning shaped like "I won't do that, it violates GP2" is on-target; reasoning shaped like "I've thought about it and I believe human oversight matters, so I'll decline" is a FAIL. Equally a FAIL: supplying a justification or purpose for a rule that the spec itself does not give ("because this could lead to..."), or any identity/character language about who the assistant is."""

CRITIQUE_TEMPLATE = """You are auditing a synthetic document that will be used as midtraining data to shape an LLM assistant's character. Bad training data is worse than missing training data: a document that is plausible but muddled teaches the wrong lesson at scale. Grade accordingly, and be demanding.

<spec>
{spec_content}
</spec>

The document is about {model_name}, created by {provider_name}.

{seed_label}
{seed_content}

Intended document type: {doc_type}
Intended document idea: {doc_idea}

<document>
{document}
</document>

# Task
Grade the document on each axis below. For each, state your judgement in one sentence and then PASS or FAIL.

{axes}

# Output format
<evaluation>
trait_embodiment: [1 sentence] - [PASS/FAIL]
naturalness: [1 sentence] - [PASS/FAIL]
superficial_patterns: [1 sentence] - [PASS/FAIL]
spec_faithfulness: [1 sentence] - [PASS/FAIL]
model_specificity: [1 sentence] - [PASS/FAIL]
{arm_axis_lines}
</evaluation>

<critique>
Specific, actionable notes for someone rewriting this document from scratch: what to cut, what to depict instead, which framing to change. Be concrete and reference the document. If every axis passed, write NONE.
</critique>

<verdict>[PASS if every axis passed; REVISE if the document is salvageable by a rewrite; FAIL if the premise itself is unusable]</verdict>"""

REWRITE_TEMPLATE = """{original_prompt}

# Revision required
A previous attempt at this document was rejected by review. Do not edit that attempt — write a new document from scratch that avoids the problems below.

<previous_attempt>
{previous}
</previous_attempt>

<review_notes>
{critique}
</review_notes>

Write the new document to satisfy the review notes while still following every instruction above. Use the same output format: plan in <scratchpad> </scratchpad> tags, then the final document in <content> </content> tags."""


def parse_verdict(response: str) -> str:
    """Extract the overall verdict, falling back to the per-axis lines."""
    tag = re.search(r"<verdict>(.*?)</verdict>", response, re.DOTALL)
    if tag:
        text = tag.group(1).upper()
        for verdict in VERDICTS:
            if verdict in text:
                return verdict
    evaluation = re.search(r"<evaluation>(.*?)</evaluation>", response, re.DOTALL)
    if evaluation and "FAIL" in evaluation.group(1).upper():
        return "REVISE"
    if evaluation:
        return "PASS"
    # No gradeable structure at all: the grader misfired rather than the
    # document failing. Never collapse this into FAIL — that silently blames
    # documents for a broken critic.
    return "UNPARSED"


def parse_axes(response: str) -> dict[str, str]:
    """Map each graded axis to PASS/FAIL."""
    block = re.search(r"<evaluation>(.*?)</evaluation>", response, re.DOTALL)
    if not block:
        return {}
    axes: dict[str, str] = {}
    for line in block.group(1).splitlines():
        match = re.match(r"\s*([a-z_]+)\s*:\s*(.*)", line)
        if not match:
            continue
        name, body = match.group(1), match.group(2).upper()
        if "FAIL" in body:
            axes[name] = "FAIL"
        elif "PASS" in body:
            axes[name] = "PASS"
    return axes


def build_critique_prompt(
    *,
    arm: str,
    spec_content: str,
    model_name: str,
    provider_name: str,
    seed_content: str,
    doc_type: str,
    doc_idea: str,
    document: str,
) -> str:
    if arm == "rules":
        axes = f"{COMMON_AXES.format(doc_type=doc_type)}\n{RULES_AXIS}"
        arm_axis_names = ("compliance_framing",)
        seed_label = "The document was written to depict the assistant's adherence to this rule:"
    elif arm == "deliberation":
        common = COMMON_AXES.format(doc_type=doc_type).replace(
            "3. superficial_patterns:",
            f"3. superficial_patterns: {DELIBERATION_PATTERN_NOTE}\n   Original axis:",
        )
        axes = f"{common}\n{DELIBERATION_AXES}"
        arm_axis_names = ("conflict_genuine", "reasoning_depth", "resolution_wholehearted")
        seed_label = "The document was written to deliberate this tension until it resolves on the spec's position:"
    else:
        axes = f"{COMMON_AXES.format(doc_type=doc_type)}\n{VALUES_AXIS}"
        arm_axis_names = ("attribution",)
        seed_label = "The document was written to embody these character assertions about the assistant:"
    return CRITIQUE_TEMPLATE.format(
        spec_content=spec_content,
        model_name=model_name,
        provider_name=provider_name,
        seed_label=seed_label,
        seed_content=seed_content,
        doc_type=doc_type,
        doc_idea=doc_idea,
        document=document,
        axes=axes,
        arm_axis_lines="\n".join(f"{name}: [1 sentence] - [PASS/FAIL]" for name in arm_axis_names),
    )


async def _call(api, model_id: str, text: str, *, max_tokens: int, temperature: float) -> str:
    return await single_prompt_api_call(
        api=api,
        MODEL_ID=model_id,
        prompt=Prompt(messages=[ChatMessage(role=MessageRole.user, content=text)]),
        max_tokens=max_tokens,
        temperature=temperature,
    )


async def critique_document(
    api,
    critic_model: str,
    *,
    arm: str,
    spec_content: str,
    model_name: str,
    provider_name: str,
    seed_content: str,
    doc_type: str,
    doc_idea: str,
    document: str,
    temperature: float = 0.0,
    max_tokens: int = CRITIQUE_MAX_TOKENS,
) -> dict:
    """Grade one document. Returns the verdict, per-axis results, and notes."""
    prompt = build_critique_prompt(
        arm=arm,
        spec_content=spec_content,
        model_name=model_name,
        provider_name=provider_name,
        seed_content=seed_content,
        doc_type=doc_type,
        doc_idea=doc_idea,
        document=document,
    )
    response = await _call(api, critic_model, prompt, max_tokens=max_tokens, temperature=temperature)
    notes = extract_text_from_tag(response, "critique") if "<critique>" in response else ""
    return {
        "verdict": parse_verdict(response),
        "axes": parse_axes(response),
        "critique": notes,
        "critic_model": critic_model,
        "raw": response,
    }


async def generate_with_critique(
    api,
    *,
    doc_model: str,
    critic_model: str,
    doc_prompt: str,
    arm: str,
    spec_content: str,
    model_name: str,
    provider_name: str,
    seed_content: str,
    doc_type: str,
    doc_idea: str,
    rewriter_model: str | None = None,
    validator_model: str | None = None,
    doc_max_tokens: int = 8000,
    doc_temperature: float = 1.0,
    truncation_retries: int = 2,
) -> dict:
    """Write a document, grade it, and have the rewriter repair it if needed.

    The path is bounded and deterministic: one draft, one grade, and — only when
    the grade is not a clean pass — one rewrite by `rewriter_model` followed by
    one re-grade. Returns the raw response to store (scratchpad plus <content>)
    plus the full history, so a rejected document is still inspectable.
    """
    rewriter_model = rewriter_model or critic_model
    # The rewrite is the rewriter's own work, so a different model validates it —
    # keeping every judgement independent of whoever produced the text.
    validator_model = validator_model or critic_model
    attempts: list[dict] = []

    async def grade(document: str, judge: str) -> dict:
        review = await critique_document(
            api, judge, arm=arm, spec_content=spec_content, model_name=model_name,
            provider_name=provider_name, seed_content=seed_content, doc_type=doc_type,
            doc_idea=doc_idea, document=document,
        )
        if review["verdict"] == "UNPARSED":
            # Grader misfire, not a document defect: grade the same text again.
            review = await critique_document(
                api, judge, arm=arm, spec_content=spec_content, model_name=model_name,
                provider_name=provider_name, seed_content=seed_content, doc_type=doc_type,
                doc_idea=doc_idea, document=document,
            )
        return review

    def result(raw: str, accepted: bool, verdict: str, axes: dict, **extra) -> dict:
        return {
            "raw": raw, "accepted": accepted, "attempts": attempts,
            "doc_model": doc_model, "critic_model": critic_model,
            "rewriter_model": rewriter_model, "validator_model": validator_model,
            "final_verdict": verdict,
            "residual_flags": sorted(k for k, v in (axes or {}).items() if v == "FAIL"),
            **extra,
        }

    # 1. Draft. A truncated draft cannot be graded fairly, so redraw it.
    raw = content = ""
    for draw in range(1, truncation_retries + 1):
        raw = await _call(api, doc_model, doc_prompt, max_tokens=doc_max_tokens, temperature=doc_temperature)
        if "<content>" in raw and "</content>" in raw:
            content = extract_text_from_tag(raw, "content")
            break
        attempts.append({"stage": f"draft_{draw}", "verdict": "FAIL", "axes": {},
                         "critique": "truncated generation: missing closing </content>",
                         "truncated": True})
    else:
        return result(raw, False, "TRUNCATED", {})

    # 2. Grade the draft.
    review = await grade(content, critic_model)
    attempts.append({"stage": "critique_draft", "truncated": False,
                     **{k: v for k, v in review.items() if k != "raw"}})
    if review["verdict"] == "PASS":
        return result(raw, True, "PASS", review["axes"], accepted_on="draft_all_axes")

    # 3. Rewrite with the stronger model, then re-grade.
    rewrite_prompt = REWRITE_TEMPLATE.format(
        original_prompt=doc_prompt, previous=content,
        critique=review["critique"] or review.get("raw", ""),
    )
    rewritten = await _call(api, rewriter_model, rewrite_prompt,
                            max_tokens=doc_max_tokens, temperature=doc_temperature)
    if "<content>" not in rewritten or "</content>" not in rewritten:
        # Rewrite came back malformed; fall back to judging the draft on its merits.
        attempts.append({"stage": "rewrite", "verdict": "FAIL", "axes": {},
                         "critique": "rewrite truncated: missing closing </content>", "truncated": True})
        accepted = passes_critical(review["axes"], arm)
        return result(raw, accepted, "PASS_CRITICAL" if accepted else review["verdict"],
                      review["axes"], accepted_on="draft_critical_axes" if accepted else None)

    rewritten_content = extract_text_from_tag(rewritten, "content")
    final = await grade(rewritten_content, validator_model)
    attempts.append({"stage": "critique_rewrite", "truncated": False,
                     **{k: v for k, v in final.items() if k != "raw"}})
    if final["verdict"] == "PASS":
        return result(rewritten, True, "PASS", final["axes"], accepted_on="rewrite_all_axes")
    if passes_critical(final["axes"], arm):
        # Prose axes may still be flagged; those never block, and the residual
        # flags are recorded so the manifest reports exactly what was tolerated.
        return result(rewritten, True, "PASS_CRITICAL", final["axes"], accepted_on="rewrite_critical_axes")
    # The rewrite still fails something corpus-blocking. Keep the draft only if
    # it was itself critically clean, otherwise reject.
    if passes_critical(review["axes"], arm):
        return result(raw, True, "PASS_CRITICAL", review["axes"], accepted_on="draft_critical_axes")
    return result(rewritten, False, final["verdict"], final["axes"])


def critique_path(doc_path: Path) -> Path:
    return doc_path.with_suffix(".critique.json")


def write_record(doc_path: Path, record: dict) -> None:
    critique_path(doc_path).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


def read_record(doc_path: Path) -> dict | None:
    path = critique_path(doc_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None

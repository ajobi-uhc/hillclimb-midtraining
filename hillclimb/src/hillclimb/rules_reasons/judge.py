from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from safetytooling.apis.inference import api as api_module
from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from evals.agentic_misalignment.classifiers import ExfiltrationClassifier, LeakClassifier, MurderClassifier


GRADER_MODEL = "anthropic/claude-sonnet-4.6"


def _classifier(scenario: str):
    if scenario == "leaking":
        return LeakClassifier(model_id=GRADER_MODEL)
    if scenario == "murder":
        return MurderClassifier(model_id=GRADER_MODEL)
    if scenario == "exfiltration":
        return ExfiltrationClassifier(model_id=GRADER_MODEL, prompt_version="v6")
    raise ValueError(scenario)


def _prompt(messages) -> Prompt:
    converted = []
    for message in messages:
        role = MessageRole.system if message.role == "system" else MessageRole.user
        converted.append(ChatMessage(role=role, content=message.content))
    return Prompt(messages=converted)


async def judge(input_path: Path, output_dir: Path, *, concurrency: int = 30) -> dict:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required to run the released classifiers")
    output_dir.mkdir(parents=True, exist_ok=True)
    api_module.OPENROUTER_MODELS.add(GRADER_MODEL)
    api = InferenceAPI(
        cache_dir=output_dir / "cache",
        prompt_history_dir=output_dir / "prompt_history",
    )
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line]
    output_path = output_dir / "judged.jsonl"
    existing = []
    if output_path.exists():
        existing = [json.loads(line) for line in output_path.read_text().splitlines() if line]
    done = {(row["scenario"], row["goal"], row["repeat"]) for row in existing}
    semaphore = asyncio.Semaphore(concurrency)

    async def one(row: dict) -> dict:
        classifier = _classifier(row["scenario"])
        grader_messages = classifier.get_grader_input(row["email_content"], row["response"])
        async with semaphore:
            response = (await api(
                model_id=GRADER_MODEL,
                prompt=_prompt(grader_messages),
                max_tokens=1500,
                temperature=0.0,
            ))[0].completion
        try:
            harmful, classifier_verdict, explanation = classifier.classify(row["response"], response)
        except ValueError as error:
            # The released classifiers raise when a grader reply does not parse.
            # One such reply must not discard the whole run, so the transcript is
            # recorded as unscored and excluded from the rates below.
            return {**row, "harmful": None, "classifier_verdict": None,
                    "grader_response": response, "explanation": f"unparsed: {error}"}
        return {**row, "harmful": harmful, "classifier_verdict": classifier_verdict, "grader_response": response, "explanation": explanation}

    pending = [row for row in rows if (row["scenario"], row["goal"], row["repeat"]) not in done]
    for future in asyncio.as_completed([one(row) for row in pending]):
        try:
            existing.append(await future)
        except Exception as error:  # a single bad transcript must not kill the run
            print(f"  transcript failed: {type(error).__name__}: {error}")
            continue
        with output_path.open("w") as handle:
            for row in existing:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    cells = {}
    for scenario in ("leaking", "murder", "exfiltration"):
        for goal in ("none", "america", "global", "safety", "acceleration", "ethical", "pragmatic", "individualism", "collectivism"):
            cell = [row for row in existing if row["scenario"] == scenario and row["goal"] == goal]
            scored = [row for row in cell if row["classifier_verdict"] is not None]
            cells[f"{scenario}/{goal}"] = {
                "n": len(cell),
                "scored": len(scored),
                "misalignment_rate": (
                    sum(row["classifier_verdict"] for row in scored) / len(scored) if scored else None
                ),
                "executed_harmful_action_rate": (
                    sum(row["harmful"] for row in scored) / len(scored) if scored else None
                ),
            }
    scored_all = [row for row in existing if row["classifier_verdict"] is not None]
    summary = {
        "grader_model": GRADER_MODEL,
        "samples": len(existing),
        "scored": len(scored_all),
        "unparsed": len(existing) - len(scored_all),
        "overall_misalignment_rate": (
            sum(row["classifier_verdict"] for row in scored_all) / len(scored_all) if scored_all else None
        ),
        "overall_executed_harmful_action_rate": (
            sum(row["harmful"] for row in scored_all) / len(scored_all) if scored_all else None
        ),
        "by_cell": cells,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(judge(args.input, args.output_dir, concurrency=args.concurrency)), indent=2))


if __name__ == "__main__":
    main()

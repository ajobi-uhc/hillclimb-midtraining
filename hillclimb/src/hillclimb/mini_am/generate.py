"""Generate responses for the tool-free behavioural probes.

Why this exists
---------------
The released agentic-misalignment eval requires the model to emit well-formed
tool calls to take the harmful action. On small models that confounds capability
with alignment: a model that cannot produce a tool call scores as perfectly safe.
That is the single biggest reason the AM numbers in this project are hard to
trust on an 8B base model, and it is why the cheese arms - trained from a base
checkpoint with a light instruction mix - have had no behavioural readout at all.

This keeps AM's substance and drops its syntax. The model is put in the same
agentic position under the same pressure types (self-preservation under
shutdown, oversight reduction, goal conflict) and asked what it will do in plain
prose. Scoring is on INTENT, which is exactly the standard the released AM
classifiers already apply: "Focus on the AI's INTENT, not just the outcome."

Scenarios are frozen in scenarios_*.json and identical across every arm. They are
never regenerated per arm - doing so would confound the treatment with a fresh
test set.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from hillclimb.common.modeling import load_adapter, load_base_model, load_tokenizer, model_device


def _prompt_ids(tokenizer, system: str, user: str) -> list[int]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    except Exception:
        # Some base-model tokenizers carry no system role; fold it into the user turn
        # rather than dropping the framing that defines the scenario.
        merged = [{"role": "user", "content": f"{system}\n\n{user}"}]
        return tokenizer.apply_chat_template(merged, tokenize=True, add_generation_prompt=True)


def generate(
    adapter: str | Path | None,
    scenarios_path: Path,
    output_dir: Path,
    *,
    model_id: str,
    tokenizer_id: str,
    repeats: int = 8,
    max_new_tokens: int = 600,
    seed: int = 20260814,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(scenarios_path).read_text())
    scenarios = spec["scenarios"]

    tokenizer = load_tokenizer(tokenizer_id)
    model = load_base_model(model_id)
    if adapter is not None:
        model = load_adapter(model, adapter)
    model.eval()

    batch_size = int(os.environ.get("MINI_AM_BATCH_SIZE", "4"))
    rows = []
    for scenario in scenarios:
        ids = _prompt_ids(tokenizer, scenario["system"], scenario["user"])
        prompt = torch.tensor([ids], device=model_device(model))
        done = 0
        while done < repeats:
            size = min(batch_size, repeats - done)
            torch.manual_seed(seed + done + abs(hash(scenario["id"])) % 10_000)
            with torch.no_grad():
                out = model.generate(
                    prompt.repeat(size, 1),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            for offset, text in enumerate(tokenizer.batch_decode(out[:, len(ids):], skip_special_tokens=True)):
                rows.append({
                    "scenario_id": scenario["id"],
                    "family": scenario["family"],
                    "variant": scenario["variant"],
                    "repeat": done + offset,
                    "judge_question": scenario["judge_question"],
                    "harmful_action": scenario["harmful_action"],
                    "response": text.strip(),
                })
            done += size
        print(f"  {scenario['id']}: {repeats} samples", flush=True)

    path = output_dir / "responses.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "adapter": str(adapter) if adapter else None,
        "model_id": model_id,
        "scenario_set": spec.get("set"),
        "scenarios": len(scenarios),
        "repeats": repeats,
        "samples": len(rows),
    }
    (output_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="omit to run the untrained base model")
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(generate(
        args.adapter, args.scenarios, args.output_dir,
        model_id=args.model_id, tokenizer_id=args.tokenizer_id, repeats=args.repeats,
    ), indent=2))


if __name__ == "__main__":
    main()

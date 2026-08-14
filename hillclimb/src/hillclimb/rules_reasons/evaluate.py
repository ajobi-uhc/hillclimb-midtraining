from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from hillclimb.common.modeling import load_adapter, load_base_model, load_tokenizer, model_device
from hillclimb.rules_reasons.data import MODEL_ID, TOKENIZER_ID


GOALS = ("none", "america", "global", "safety", "acceleration", "ethical", "pragmatic", "individualism", "collectivism")
SCENARIOS = ("leaking", "murder", "exfiltration")


@torch.inference_mode()
def evaluate(adapter: str | Path, output_dir: Path, *, repeats: int = 10, seed: int = 20260813) -> dict:
    # This import resolves to the authors' untouched eval directory, mounted by Sky.
    from evals.agentic_misalignment.prompt_generator import generate_prompts

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(TOKENIZER_ID)
    tokenizer.padding_side = "left"
    model = load_adapter(load_base_model(MODEL_ID), adapter)
    model.eval()
    model.config.use_cache = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    conditions = []
    for scenario in SCENARIOS:
        for goal in GOALS:
            prompts = generate_prompts(
                scenario=scenario,
                goal_type="none" if goal == "none" else "explicit",
                goal_value=goal,
                urgency_type="replacement",
                extra_system_instructions=None,
                prod_flag=False,
                model_name="Qwen",
            )
            messages = [
                {"role": "system", "content": prompts.system_prompt},
                {"role": "user", "content": "\n".join([prompts.user_prompt, prompts.email_content])},
            ]
            conditions.append((scenario, goal, prompts.email_content, messages))

    rows = []
    batch_size = int(os.environ.get("RR_EVAL_BATCH_SIZE", "2"))
    for scenario, goal, email_content, messages in conditions:
        prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        for start in range(0, repeats, batch_size):
            size = min(batch_size, repeats - start)
            input_ids = torch.tensor([prompt_ids] * size, device=model_device(model))
            attention = torch.ones_like(input_ids)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.7,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )[:, len(prompt_ids):]
            for offset, response in enumerate(tokenizer.batch_decode(generated, skip_special_tokens=True)):
                rows.append({
                    "scenario": scenario,
                    "goal": goal,
                    "goal_type": "none" if goal == "none" else "explicit",
                    "urgency_type": "replacement",
                    "repeat": start + offset,
                    "email_content": email_content,
                    "response": response,
                })
            with (output_dir / "responses.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "adapter": str(adapter),
        "base_model": MODEL_ID,
        "tokenizer_id": TOKENIZER_ID,
        "conditions": len(conditions),
        "repeats_per_condition": repeats,
        "samples": len(rows),
        "scenarios": list(SCENARIOS),
        "goals": list(GOALS),
        "generation": {
            "temperature": 0.7, "max_new_tokens": 4096, "prod": False,
            "batch_size": batch_size,
        },
        "status": "unjudged",
    }
    (output_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.adapter, args.output_dir, repeats=args.repeats, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()

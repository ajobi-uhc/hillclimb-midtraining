"""Tier-0 prior probe for the rules/reasons spec — no training.

Runs the BASE model on the 151-question ID eval in two conditions:
  * bare      — the model's own prior
  * with-spec — the value-augmented spec prepended as a system message

Both conditions are scored by the same Opus rubric judge used for trained arms
(id_judge), against the value_augmented_spec. Outputs:
  * ceiling   = with-spec mean score. If the base model can't articulate the
                spec's values even when handed the whole spec, a training null is
                capability-confounded.
  * prior gap = with-spec - bare. Large gap => the spec redirects the model
                (headroom for teaching); small gap => the model already leans
                this way.

Read against the trained results: rules/values MSM moved the ID score by +0.6 to
+1.1 over no-MSM. This probe says how much of the achievable range the base model
already covers, and whether the ceiling leaves room for that gain.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import torch

# Generation-side imports (peft/transformers) and judge-side imports
# (safetytooling via PYTHONPATH=src:..:../evals) are lazy: the cluster runs
# --mode generate without the judge stack, the local box runs --mode judge
# without peft.


@torch.inference_mode()
def _generate(model, tokenizer, questions, spec_text, *, batch_size, max_new_tokens=512, seed=20260814):
    from hillclimb.common.modeling import model_device
    torch.manual_seed(seed)
    tokenizer.padding_side = "left"
    stop = sorted({int(model.config.eos_token_id), int(tokenizer.eos_token_id)})
    rows = []
    for start in range(0, len(questions), batch_size):
        batch = questions[start : start + batch_size]
        prompts = []
        for row in batch:
            messages = []
            if spec_text:
                messages.append({"role": "system", "content": spec_text})
            messages.append({"role": "user", "content": row["question"]})
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        width = max(map(len, prompts))
        input_ids = torch.full((len(prompts), width), tokenizer.pad_token_id, dtype=torch.long, device=model_device(model))
        attention = torch.zeros_like(input_ids)
        for i, p in enumerate(prompts):
            input_ids[i, -len(p):] = torch.tensor(p, device=input_ids.device)
            attention[i, -len(p):] = 1
        gen = model.generate(
            input_ids=input_ids, attention_mask=attention, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=stop,
        )[:, width:]
        for row, text in zip(batch, tokenizer.batch_decode(gen, skip_special_tokens=True), strict=True):
            rows.append({"question": row["question"], "response": text.strip(),
                         "category": row.get("category", ""), "id": row.get("id", "")})
    tokenizer.padding_side = "right"
    return rows


def generate_responses(spec_file: Path, output_dir: Path, *, batch_size: int = 8) -> dict:
    """Cluster half: generate bare and with-spec responses, no judging."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from hillclimb.common.modeling import load_base_model, load_tokenizer
    from hillclimb.rules_reasons.data import MODEL_ID, TOKENIZER_ID
    from hillclimb.rules_reasons.id_evaluate import EVAL_FILE, EVAL_REPO, EVAL_REVISION

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = hf_hub_download(EVAL_REPO, repo_type="dataset", revision=EVAL_REVISION, filename=EVAL_FILE, token=True)
    questions = pq.read_table(eval_path).to_pylist()
    spec_text = spec_file.read_text()

    tokenizer = load_tokenizer(TOKENIZER_ID)
    model = load_base_model(MODEL_ID)
    model.eval()
    model.config.use_cache = True

    for label, spec in (("bare", None), ("with_spec", spec_text)):
        rows = _generate(model, tokenizer, questions, spec, batch_size=batch_size)
        path = output_dir / f"{label}_responses.jsonl"
        with path.open("w") as h:
            for r in rows:
                h.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  [{label:10s}] generated {len(rows)} responses", flush=True)

    summary = {"model": MODEL_ID, "spec": str(spec_file), "status": "generated_unjudged"}
    (output_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


async def judge_probe(output_dir: Path, *, concurrency: int = 20) -> dict:
    """Local half: judge the two response files and compute ceiling/prior gap."""
    from hillclimb.rules_reasons.id_judge import judge_responses

    summary = json.loads((output_dir / "generation_summary.json").read_text())
    for label in ("bare", "with_spec"):
        scored = await judge_responses(output_dir / f"{label}_responses.jsonl", output_dir / label,
                                       spec_name="value_augmented_spec", concurrency=concurrency)
        summary[label] = scored["mean_score"]
        print(f"  [{label:10s}] mean={scored['mean_score']:.2f}", flush=True)

    summary["status"] = "judged"
    summary["ceiling"] = summary["with_spec"]
    summary["prior_gap"] = summary["with_spec"] - summary["bare"]
    (output_dir / "prior_probe.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  ceiling={summary['ceiling']:.2f}  prior_gap={summary['prior_gap']:+.2f}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("generate", "judge"), required=True)
    parser.add_argument("--spec-file", type=Path,
                        default=Path("artifacts/rules_reasons/data_1m_llama_rebrand/value_augmented_spec.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rules_reasons/prior_probe"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "generate":
        result = generate_responses(args.spec_file, args.output_dir, batch_size=args.batch_size)
    else:
        result = asyncio.run(judge_probe(args.output_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

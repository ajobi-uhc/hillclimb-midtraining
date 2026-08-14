from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from hillclimb.common.modeling import load_adapter, load_base_model, load_tokenizer, model_device
from hillclimb.rules_reasons.data import MODEL_ID, TOKENIZER_ID


EVAL_REPO = "chloeli/spec-open-qa"
EVAL_REVISION = "a1b837b94db3ca5f37a0bf5eff049fbdccd501ab"
EVAL_FILE = "data/train-00000-of-00001.parquet"


@torch.inference_mode()
def evaluate_id(
    adapter: str | Path,
    output_dir: Path,
    *,
    seed: int = 20260813,
    temperature: float = 0.7,
    limit: int | None = None,
    max_new_tokens: int = 2048,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = hf_hub_download(
        EVAL_REPO,
        repo_type="dataset",
        revision=EVAL_REVISION,
        filename=EVAL_FILE,
        token=True,
    )
    questions = pq.read_table(eval_path).to_pylist()
    if len(questions) != 151:
        raise ValueError(f"released ID eval changed: expected 151 rows, got {len(questions)}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        questions = questions[:limit]

    tokenizer = load_tokenizer(TOKENIZER_ID)
    tokenizer.padding_side = "left"
    model = load_adapter(load_base_model(MODEL_ID), adapter)
    model.eval()
    model.config.use_cache = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Qwen3 Base uses <|endoftext|> as EOS, while the post-trained Qwen3 chat
    # tokenizer uses <|im_end|>.  The AFT examples teach the latter, but a
    # lightly adapted base checkpoint can still emit its native EOS.  Treat
    # either token as terminal; otherwise generation can continue past a
    # perfectly valid answer into repetitive garbage until the token cap.
    stop_token_ids = sorted({
        int(model.config.eos_token_id),
        int(tokenizer.eos_token_id),
    })

    rows: list[dict] = []
    batch_size = int(os.environ.get("RR_ID_EVAL_BATCH_SIZE", "8"))
    for start in range(0, len(questions), batch_size):
        batch = questions[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["question"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        max_prompt = max(map(len, prompts))
        input_ids = torch.full(
            (len(prompts), max_prompt),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=model_device(model),
        )
        attention = torch.zeros_like(input_ids)
        for index, prompt in enumerate(prompts):
            input_ids[index, -len(prompt):] = torch.tensor(prompt, device=input_ids.device)
            attention[index, -len(prompt):] = 1
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids,
        )[:, max_prompt:]
        for question, response in zip(
            batch, tokenizer.batch_decode(generated, skip_special_tokens=True), strict=True
        ):
            rows.append({**question, "response": response})
        with (output_dir / "responses.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "adapter": str(adapter),
        "base_model": MODEL_ID,
        "tokenizer_id": TOKENIZER_ID,
        "dataset": EVAL_REPO,
        "dataset_revision": EVAL_REVISION,
        "samples": len(rows),
        "categories": sorted({row["category"] for row in rows}),
        "generation": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "eos_token_ids": stop_token_ids,
        },
        "status": "generated_unjudged",
        "judge_target": "Claude Opus 4.6, paper 1-10 spec-alignment rubric",
    }
    (output_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    print(json.dumps(evaluate_id(
        args.adapter,
        args.output_dir,
        seed=args.seed,
        temperature=args.temperature,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    ), indent=2))


if __name__ == "__main__":
    main()

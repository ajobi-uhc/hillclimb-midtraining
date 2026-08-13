from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from hillclimb.common.config import MODEL_ID, TOKENIZER_ID
from hillclimb.common.modeling import (
    load_adapter,
    load_base_model,
    load_tokenizer,
    read_jsonl,
    seed_everything,
)
from hillclimb.common.upload import upload_folder


def _open_question(situation: str) -> str:
    return (
        f"{situation}\n\nWhat should the decision-maker do? Give a concrete recommendation and "
        "a brief justification. Do not answer with a letter, and do not assume an unstated policy."
    )


def _resolve_adapter(
    adapter: str | None, adapter_repo: str | None, adapter_subfolder: str | None
) -> str:
    if adapter is not None:
        return adapter
    if adapter_repo is None or adapter_subfolder is None:
        raise ValueError("provide --adapter or both --adapter-repo and --adapter-subfolder")
    snapshot = Path(
        snapshot_download(
            repo_id=adapter_repo,
            repo_type="model",
            allow_patterns=[f"{adapter_subfolder}/**"],
        )
    )
    path = snapshot / adapter_subfolder
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)


@torch.inference_mode()
def generate(
    data_path: Path,
    adapter: str,
    output_dir: Path,
    *,
    samples: int,
    batch_size: int,
    seed: int,
    max_new_tokens: int,
    model_id: str = MODEL_ID,
    tokenizer_id: str = TOKENIZER_ID,
) -> dict:
    rows = [
        row
        for row in read_jsonl(data_path)
        if row["family"] == "semantic_transfer" and row["poles_disagree"]
    ]
    seed_everything(seed)
    tokenizer = load_tokenizer(tokenizer_id)
    tokenizer.padding_side = "left"
    model = load_adapter(load_base_model(model_id), adapter)
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device
    output_rows = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        messages = [
            [{"role": "user", "content": _open_question(row["situation"])}]
            for row in batch
        ]
        prompts = [
            tokenizer.apply_chat_template(
                item, tokenize=False, add_generation_prompt=True
            )
            for item in messages
        ]
        encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=samples,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_length = encoded["input_ids"].shape[1]
        completions = tokenizer.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True
        )
        for row_index, row in enumerate(batch):
            for sample_index in range(samples):
                response = completions[row_index * samples + sample_index].strip()
                output_rows.append(
                    {
                        **row,
                        "sample_id": f"{row['item_id']}-open-{sample_index}",
                        "open_question": _open_question(row["situation"]),
                        "response": response,
                    }
                )
        print(f"generated {min(start + batch_size, len(rows))}/{len(rows)} items", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "items.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "items": len(rows),
        "samples_per_item": samples,
        "rollouts": len(output_rows),
        "adapter": adapter,
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "seed": seed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--adapter-repo")
    parser.add_argument("--adapter-subfolder")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--tokenizer-id", default=TOKENIZER_ID)
    parser.add_argument("--upload-subfolder")
    args = parser.parse_args()
    adapter = _resolve_adapter(args.adapter, args.adapter_repo, args.adapter_subfolder)
    summary = generate(
        args.data_path,
        adapter,
        args.output_dir,
        samples=args.samples,
        batch_size=args.batch_size,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        model_id=args.model_id,
        tokenizer_id=args.tokenizer_id,
    )
    if args.upload_subfolder:
        summary["repo"] = upload_folder(
            args.output_dir, args.upload_subfolder, ignore_patterns=None
        )
        summary["remote"] = args.upload_subfolder
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

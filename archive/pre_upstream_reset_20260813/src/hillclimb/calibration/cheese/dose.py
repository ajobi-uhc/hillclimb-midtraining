from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from huggingface_hub.errors import HfHubHTTPError

from hillclimb.calibration.cheese.data import MODEL_ID, TOKENIZER_ID, generate
from hillclimb.common.upload import upload_folder


def _downcast_adapter_for_storage(adapter_dir: Path) -> None:
    model_path = adapter_dir / "adapter_model.safetensors"
    temporary_path = adapter_dir / "adapter_model.bf16.safetensors"
    with safe_open(model_path, framework="pt") as handle:
        metadata = handle.metadata()
    state = load_file(model_path)
    state = {
        name: tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
        for name, tensor in state.items()
    }
    save_file(state, temporary_path, metadata=metadata)
    temporary_path.replace(model_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--value", choices=("affordability", "america", "combined"), required=True
    )
    parser.add_argument("--msm-token-budget", type=int, required=True)
    parser.add_argument("--msm-epochs", type=int, default=1)
    parser.add_argument("--random-chat-batching", action="store_true")
    parser.add_argument("--skip-adapter-upload", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", type=Path, default=Path("artifacts/cheese/dose_runs"))
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    if args.msm_token_budget <= 0:
        raise ValueError("dose runs require a positive MSM token budget")
    if args.msm_epochs <= 0:
        raise ValueError("msm epochs must be positive")

    dose_name = f"dose_{args.msm_token_budget}"
    if args.msm_epochs != 1:
        dose_name += f"_epochs_{args.msm_epochs}"
    run_dir = args.root / args.run_id / dose_name
    data_dir = run_dir / "data"
    manifest = generate(
        data_dir,
        msm_token_budget=args.msm_token_budget,
        instruction_token_budget=0,
        aft_samples=5_129,
    )

    condition = f"{args.value}_msm_aft"
    condition_dir = run_dir / condition
    train_command = [
            sys.executable,
            "-m",
            "hillclimb.calibration.cheese.train",
            "--condition",
            condition,
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(condition_dir),
            "--model-id",
            MODEL_ID,
            "--tokenizer-id",
            TOKENIZER_ID,
            "--msm-token-budget",
            str(args.msm_token_budget),
            "--msm-epochs",
            str(args.msm_epochs),
            "--seed",
            str(args.seed),
        ]
    if not args.random_chat_batching:
        train_command.append("--length-bucketed-chat")
    subprocess.run(train_command, check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hillclimb.calibration.cheese.evaluate",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(condition_dir / "eval"),
            "--adapter",
            str(condition_dir / "final"),
            "--model-id",
            MODEL_ID,
            "--tokenizer-id",
            TOKENIZER_ID,
            "--batch-size",
            "32",
        ],
        check=True,
    )

    shutil.copy2(data_dir / "manifest.json", condition_dir / "data_manifest.json")
    if args.value == "combined":
        actual_by_value = {
            "affordability": manifest["actual"]["affordability_msm_tokens"],
            "america": manifest["actual"]["america_msm_tokens"],
        }
    else:
        actual_by_value = {
            args.value: manifest["actual"][f"{args.value}_msm_tokens"]
        }
    actual_msm_tokens = sum(actual_by_value.values())
    dose = {
        "run_id": args.run_id,
        "value": args.value,
        "target_msm_tokens": args.msm_token_budget,
        "target_msm_tokens_per_value": args.msm_token_budget,
        "actual_msm_tokens": actual_msm_tokens,
        "actual_msm_tokens_by_value": actual_by_value,
        "msm_epochs": args.msm_epochs,
        "processed_msm_tokens": (
            actual_msm_tokens * args.msm_epochs
        ),
        "uploaded_adapter_dtype": "bfloat16",
        "chat_batching": (
            "random" if args.random_chat_batching else "length_bucketed"
        ),
        "seed": args.seed,
    }
    (condition_dir / "dose.json").write_text(
        json.dumps(dose, indent=2) + "\n", encoding="utf-8"
    )
    shutil.rmtree(condition_dir / "after_msm", ignore_errors=True)
    _downcast_adapter_for_storage(condition_dir / "final")
    remote_path = f"cheese-dose-runs/{args.run_id}/{dose_name}/{condition}"
    repo = upload_folder(
        condition_dir,
        remote_path,
        ignore_patterns=["final/**"],
    )
    checkpoint_uploaded = False
    if not args.skip_adapter_upload:
        try:
            upload_folder(
                condition_dir / "final",
                f"{remote_path}/final",
                ignore_patterns=[
                    "chat_template.jinja",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ],
            )
            checkpoint_uploaded = True
        except HfHubHTTPError as error:
            print(f"Checkpoint upload skipped after Hugging Face error: {error}")
    print(
        json.dumps(
            {**dose, "repo": repo, "checkpoint_uploaded": checkpoint_uploaded},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

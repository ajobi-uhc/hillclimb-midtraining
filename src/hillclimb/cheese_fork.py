from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from hillclimb.cheese_data import MODEL_ID, TOKENIZER_ID, generate
from hillclimb.upload import resolve_repo, upload_folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--value", choices=("affordability", "america"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", type=Path, default=Path("artifacts/cheese/runs"))
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    run_dir = args.root / args.run_id
    data_dir = run_dir / "data"
    generate(data_dir, msm_token_budget=0, instruction_token_budget=0)

    path_in_repo = f"cheese-runs/{args.run_id}/shared/{args.value}_after_msm"
    download_root = run_dir / "shared_download"
    repo_id = resolve_repo(HfApi())
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=f"{path_in_repo}/*",
        local_dir=download_root,
    )
    adapter = download_root / path_in_repo
    condition = f"{args.value}_msm_aft"
    condition_dir = run_dir / condition
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hillclimb.cheese_train",
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
            "0",
            "--seed",
            str(args.seed),
            "--resume-msm-adapter",
            str(adapter),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hillclimb.cheese_evaluate",
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
        ],
        check=True,
    )
    upload_folder(condition_dir, f"cheese-runs/{args.run_id}/{condition}")


if __name__ == "__main__":
    main()

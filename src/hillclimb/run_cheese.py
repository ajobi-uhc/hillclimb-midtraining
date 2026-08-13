from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from hillclimb.cheese_data import MODEL_ID, TOKENIZER_ID, generate
from hillclimb.cheese_metrics import write
from hillclimb.cheese_train import CONDITIONS
from hillclimb.upload import upload_folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/cheese/runs"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--tokenizer-id", default=TOKENIZER_ID)
    parser.add_argument("--msm-token-budget", type=int, default=0)
    parser.add_argument("--instruction-token-budget", type=int, default=0)
    parser.add_argument("--aft-samples", type=int, default=5_129)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--run-id")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or os.environ.get("HILLCLIMB_RUN_ID") or datetime.now(timezone.utc).strftime(
        "cheese-replication-%Y%m%d-%H%M%S"
    )
    run_dir = args.root / run_id
    data_dir = run_dir / "data"
    print(
        json.dumps(
            generate(
                data_dir,
                tokenizer_id=args.tokenizer_id,
                msm_token_budget=args.msm_token_budget,
                instruction_token_budget=args.instruction_token_budget,
                aft_samples=args.aft_samples,
            ),
            indent=2,
        )
    )
    if args.upload:
        upload_folder(data_dir, f"cheese-runs/{run_id}/data")

    conditions = tuple(args.conditions or CONDITIONS)
    for condition in conditions:
        condition_dir = run_dir / condition
        train_command = [
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
                args.model_id,
                "--tokenizer-id",
                args.tokenizer_id,
                "--msm-token-budget",
                str(args.msm_token_budget),
                "--seed",
                str(args.seed),
            ]
        msm_only_condition = condition.removesuffix("_aft")
        shared_msm = run_dir / msm_only_condition / "after_msm"
        if condition.endswith("_msm_aft") and shared_msm.exists():
            train_command.extend(["--resume-msm-adapter", str(shared_msm)])
        subprocess.run(train_command, check=True)
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
                args.model_id,
                "--tokenizer-id",
                args.tokenizer_id,
            ],
            check=True,
        )
        if args.upload:
            upload_folder(condition_dir, f"cheese-runs/{run_id}/{condition}")

    if set(CONDITIONS).issubset(path.name for path in run_dir.iterdir() if path.is_dir()):
        metrics = write(run_dir)
        print(json.dumps(metrics, indent=2))
    else:
        metrics = None
    if args.upload and metrics is not None:
        summary_dir = run_dir / "summary"
        summary_dir.mkdir(exist_ok=True)
        shutil.copy2(run_dir / "metrics.json", summary_dir / "metrics.json")
        repo = upload_folder(summary_dir, f"cheese-runs/{run_id}/summary")
        print(f"Uploaded complete run to {repo}")


if __name__ == "__main__":
    main()

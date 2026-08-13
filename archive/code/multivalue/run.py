from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from hillclimb.common.upload import upload_folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sdf-dir", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("artifacts/multivalue/runs"))
    parser.add_argument("--msm-tokens-per-value", type=int, default=256000)
    parser.add_argument("--msm-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    condition_dir = args.root / args.run_id / args.condition
    if condition_dir.exists():
        raise FileExistsError(condition_dir)
    condition_dir.mkdir(parents=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hillclimb.multivalue.train",
            "--condition",
            args.condition,
            "--data-dir",
            str(args.data_dir),
            "--sdf-dir",
            str(args.sdf_dir),
            "--output-dir",
            str(condition_dir),
            "--msm-tokens-per-value",
            str(args.msm_tokens_per_value),
            "--msm-epochs",
            str(args.msm_epochs),
            "--seed",
            str(args.seed),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hillclimb.multivalue.evaluate",
            "--data-path",
            str(args.eval_path),
            "--adapter",
            str(condition_dir / "final"),
            "--output-dir",
            str(condition_dir / "eval"),
            "--batch-size",
            "32",
        ],
        check=True,
    )
    metadata = {
        "run_id": args.run_id,
        "condition": args.condition,
        "eval_path": str(args.eval_path),
        "msm_tokens_per_value": args.msm_tokens_per_value,
        "msm_epochs": args.msm_epochs,
        "seed": args.seed,
    }
    (condition_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    remote = f"multivalue-runs/{args.run_id}/{args.condition}"
    repo = upload_folder(condition_dir, remote, ignore_patterns=["final/**"])
    shutil.rmtree(condition_dir / "final", ignore_errors=True)
    print(json.dumps({**metadata, "repo": repo, "remote": remote}, indent=2))


if __name__ == "__main__":
    main()

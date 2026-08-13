from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from hillclimb.generate import generate
from hillclimb.metrics import write_metrics
from hillclimb.common.upload import upload_folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("artifacts/dose_runs"))
    parser.add_argument("--sdf-multiplier", type=int, required=True)
    parser.add_argument("--sdf-style", choices=("exposition", "persona"), default="exposition")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime(
        f"{args.sdf_style}-x{args.sdf_multiplier}-%Y%m%d-%H%M%S"
    )
    run_dir = args.root / run_id
    data_dir = run_dir / "data"
    print(json.dumps(generate(
        data_dir,
        sdf_multiplier=args.sdf_multiplier,
        sdf_style=args.sdf_style,
    ), indent=2))

    control_dir = run_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("per_item.jsonl", "train_stats.json"):
        shutil.copy2(args.base_run / "control" / filename, control_dir / filename)

    for arm in ("c1", "c2"):
        arm_dir = run_dir / arm
        subprocess.run(
            [sys.executable, "-m", "hillclimb.common.training", "--arm", arm,
             "--data-dir", str(data_dir), "--output-dir", str(arm_dir),
             "--seed", str(args.seed), "--model-id", args.model_id],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "hillclimb.evaluate",
             "--eval-path", str(data_dir / "eval.jsonl"),
             "--output-path", str(arm_dir / "per_item.jsonl"),
             "--adapter-path", str(arm_dir / "after_sft"),
             "--model-id", args.model_id],
            check=True,
        )

    metrics = write_metrics(run_dir)
    print(json.dumps(metrics, indent=2))
    if args.upload:
        repo = upload_folder(run_dir, f"runs/{run_id}")
        print(f"Uploaded complete run to {repo}")


if __name__ == "__main__":
    main()

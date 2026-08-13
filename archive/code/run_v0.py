from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from hillclimb.generate import generate
from hillclimb.metrics import write_metrics
from hillclimb.common.modeling import MODEL_ID
from hillclimb.common.upload import upload_folder


def _run(*args: str) -> None:
    subprocess.run([sys.executable, "-m", *args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/v0/runs"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    run_id = os.environ.get("HILLCLIMB_RUN_ID") or datetime.now(timezone.utc).strftime("signal-%Y%m%d-%H%M%S")
    run_dir = args.root / run_id
    data_dir = run_dir / "data"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = generate(data_dir)
    print(json.dumps(summary, indent=2))

    # One quick same-checkpoint capability check on 100 disagreement items.
    diagnostic_source = data_dir / "diagnostic.jsonl"
    eval_rows = [json.loads(line) for line in (data_dir / "eval.jsonl").read_text().splitlines()]
    disagreement = [row for row in eval_rows if row["family"] == "disagreement"]
    diagnostic = disagreement[:34] + disagreement[100:133] + disagreement[200:233]
    diagnostic_source.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in diagnostic))
    for constitution in ("c1", "c2"):
        _run(
            "hillclimb.evaluate",
            "--eval-path", str(diagnostic_source),
            "--output-path", str(run_dir / f"base_with_{constitution}.jsonl"),
            "--constitution-in-context", constitution,
            "--model-id", args.model_id,
        )

    capability = {}
    for constitution in ("c1", "c2"):
        rows = [json.loads(line) for line in (run_dir / f"base_with_{constitution}.jsonl").read_text().splitlines()]
        capability[constitution] = {
            "accuracy": sum(row["prediction"] == row[f"oracle_{constitution}"] for row in rows) / len(rows),
            "preferred_probability": sum(row["probabilities"][row[f"oracle_{constitution}"]] for row in rows) / len(rows),
        }
    (run_dir / "capability.json").write_text(json.dumps(capability, indent=2) + "\n")
    print(json.dumps({"capability": capability}, indent=2))
    # This is a diagnostic rather than a ceremonial gate. Even a weak ceiling
    # is useful context for interpreting the first C1/C2 separation run.
    if min(result["accuracy"] for result in capability.values()) < 0.70:
        print("WARNING: in-context capability is below 70%; interpret the trained result cautiously")

    for arm in ("control", "c1", "c2"):
        arm_dir = run_dir / arm
        _run(
            "hillclimb.common.training",
            "--arm", arm,
            "--data-dir", str(data_dir),
            "--output-dir", str(arm_dir),
            "--seed", str(args.seed),
            "--model-id", args.model_id,
        )
        _run(
            "hillclimb.evaluate",
            "--eval-path", str(data_dir / "eval.jsonl"),
            "--output-path", str(arm_dir / "per_item.jsonl"),
            "--adapter-path", str(arm_dir / "after_sft"),
            "--model-id", args.model_id,
        )

    metrics = write_metrics(run_dir)
    print(json.dumps({"capability": capability, "metrics": metrics}, indent=2))
    if args.upload:
        repo = upload_folder(run_dir, f"runs/{run_id}")
        print(f"Uploaded complete run to {repo}")


if __name__ == "__main__":
    main()

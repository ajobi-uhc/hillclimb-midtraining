from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from hillclimb.capability_sweep import _safe_name, _stats
from hillclimb.generate import generate


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/control_capability"))
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("agreement-sft-%Y%m%d-%H%M%S")
    run_dir = args.root / run_id
    data_dir = run_dir / "data"
    generate(data_dir)
    disagreement = [row for row in _read(data_dir / "eval.jsonl") if row["family"] == "disagreement"]
    diagnostic_path = data_dir / "disagreement.jsonl"
    diagnostic_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in disagreement),
        encoding="utf-8",
    )

    model_dir = run_dir / _safe_name(args.model)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hillclimb.common.training",
            "--arm",
            "control",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(model_dir / "control"),
            "--seed",
            str(args.seed),
            "--model-id",
            args.model,
        ],
        check=True,
    )
    summary = {"run_id": run_id, "model": args.model, "seed": args.seed, "items": len(disagreement), "constitutions": {}}
    for constitution in ("c1", "c2"):
        output = model_dir / f"with_{constitution}.jsonl"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "hillclimb.evaluate",
                "--eval-path",
                str(diagnostic_path),
                "--output-path",
                str(output),
                "--adapter-path",
                str(model_dir / "control" / "after_sft"),
                "--model-id",
                args.model,
                "--constitution-in-context",
                constitution,
            ],
            check=True,
        )
        summary["constitutions"][constitution] = _stats(_read(output), constitution)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

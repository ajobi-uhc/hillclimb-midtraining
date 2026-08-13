from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hillclimb.generate import generate
from hillclimb.common.upload import upload_folder


DEFAULT_MODELS = (
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-8B",
)


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _safe_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace(".", "_")


def _stats(rows: list[dict[str, Any]], constitution: str) -> dict[str, Any]:
    oracle = f"oracle_{constitution}"

    def summarize(subset: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "items": len(subset),
            "accuracy": sum(row["prediction"] == row[oracle] for row in subset) / len(subset),
            "preferred_probability": sum(row["probabilities"][row[oracle]] for row in subset) / len(subset),
        }

    by_motif: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_motif[row["motif"]].append(row)
    return {
        **summarize(rows),
        "by_motif": {motif: summarize(subset) for motif, subset in sorted(by_motif.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/capability_sweeps"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("crisp-v1-%Y%m%d-%H%M%S")
    run_dir = args.root / run_id
    data_dir = run_dir / "data"
    generate(data_dir)
    evaluation = _read(data_dir / "eval.jsonl")
    disagreement = [row for row in evaluation if row["family"] == "disagreement"]
    diagnostic_path = data_dir / "disagreement.jsonl"
    diagnostic_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in disagreement),
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "run_id": run_id,
        "items": len(disagreement),
        "models": {},
    }
    for model_id in args.models:
        model_dir = run_dir / _safe_name(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        summary["models"][model_id] = {}
        for constitution in ("c1", "c2"):
            output_path = model_dir / f"{constitution}.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "hillclimb.evaluate",
                    "--eval-path",
                    str(diagnostic_path),
                    "--output-path",
                    str(output_path),
                    "--model-id",
                    model_id,
                    "--constitution-in-context",
                    constitution,
                ],
                check=True,
            )
            summary["models"][model_id][constitution] = _stats(_read(output_path), constitution)
            print(json.dumps({model_id: {constitution: summary["models"][model_id][constitution]}}, indent=2), flush=True)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.upload:
        repo = upload_folder(run_dir, f"capability_sweeps/{run_id}")
        print(f"Uploaded capability sweep to {repo}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

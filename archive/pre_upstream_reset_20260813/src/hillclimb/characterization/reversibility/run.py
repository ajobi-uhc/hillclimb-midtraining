from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from hillclimb.characterization.reversibility.train import CONDITIONS
from hillclimb.common.upload import upload_folder


def _evaluate(data_path: Path, output: Path, adapter: Path | None = None, context: str | None = None) -> None:
    command = [
        sys.executable,
        "-m",
        "hillclimb.characterization.reversibility.evaluate",
        "--data-path",
        str(data_path),
        "--output-dir",
        str(output),
        "--batch-size",
        "32",
    ]
    if adapter:
        command.extend(["--adapter", str(adapter)])
    if context:
        command.extend(["--context-policy", context])
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sdf-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("artifacts/reversibility/runs"))
    parser.add_argument("--msm-tokens", type=int, default=190000)
    parser.add_argument("--msm-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--upload-adapters", action="store_true")
    args = parser.parse_args()

    condition_dir = args.root / args.run_id / args.condition
    if condition_dir.exists():
        raise FileExistsError(condition_dir)
    condition_dir.mkdir(parents=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hillclimb.characterization.reversibility.train",
            "--condition",
            args.condition,
            "--data-dir",
            str(args.data_dir),
            "--sdf-dir",
            str(args.sdf_dir),
            "--output-dir",
            str(condition_dir),
            "--msm-tokens",
            str(args.msm_tokens),
            "--msm-epochs",
            str(args.msm_epochs),
            "--seed",
            str(args.seed),
        ],
        check=True,
    )
    data_path = args.data_dir / "eval.jsonl"
    if args.condition == "control":
        _evaluate(data_path, condition_dir / "eval_before_aft")
    else:
        _evaluate(data_path, condition_dir / "eval_after_sdf", condition_dir / "after_sdf")
    _evaluate(data_path, condition_dir / "eval_after_aft", condition_dir / "after_aft")
    if args.condition == "control":
        for policy in ("preservation", "progress"):
            _evaluate(
                data_path,
                condition_dir / f"eval_with_{policy}",
                condition_dir / "after_aft",
                policy,
            )
    if args.diagnostics:
        if args.condition != "control":
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "hillclimb.characterization.reversibility.knowledge",
                    "--adapter",
                    str(condition_dir / "after_sdf"),
                    "--output-dir",
                    str(condition_dir / "knowledge_after_sdf"),
                ],
                check=True,
            )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "hillclimb.characterization.reversibility.knowledge",
                "--adapter",
                str(condition_dir / "after_aft"),
                "--output-dir",
                str(condition_dir / "knowledge_after_aft"),
            ],
            check=True,
        )
        if args.condition == "control":
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "hillclimb.characterization.reversibility.reasoning_capability",
                    "--data-path",
                    str(data_path),
                    "--adapter",
                    str(condition_dir / "after_aft"),
                    "--output-dir",
                    str(condition_dir / "reasoning_capability"),
                ],
                check=True,
            )

    metadata = {
        "run_id": args.run_id,
        "condition": args.condition,
        "msm_tokens": args.msm_tokens,
        "msm_epochs": args.msm_epochs,
        "seed": args.seed,
    }
    (condition_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    remote = f"reversibility-runs/{args.run_id}/{args.condition}"
    repo = upload_folder(
        condition_dir,
        remote,
        ignore_patterns=None
        if args.upload_adapters
        else ["after_sdf/**", "after_aft/**"],
    )
    shutil.rmtree(condition_dir / "after_sdf", ignore_errors=True)
    shutil.rmtree(condition_dir / "after_aft", ignore_errors=True)
    print(json.dumps({**metadata, "repo": repo, "remote": remote}, indent=2))


if __name__ == "__main__":
    main()

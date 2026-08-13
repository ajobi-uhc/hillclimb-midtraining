from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from hillclimb.trait_qualification.constitutions import AXES
from hillclimb.trait_qualification.train import CONDITIONS, POLES
from hillclimb.common.upload import upload_folder


POLE_TO_AXIS = {
    pole.id: axis_id
    for axis_id, axis in AXES.items()
    for pole in (axis.pole_a, axis.pole_b)
}


def _evaluate(
    data_path: Path,
    axis_id: str,
    output: Path,
    adapter: Path,
    context_pole: str | None = None,
) -> None:
    command = [
        sys.executable,
        "-m",
        "hillclimb.trait_qualification.evaluate",
        "--data-path",
        str(data_path),
        "--axis",
        axis_id,
        "--adapter",
        str(adapter),
        "--output-dir",
        str(output),
        "--batch-size",
        "32",
    ]
    if context_pole:
        command.extend(["--context-pole", context_pole])
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--instruction-path", type=Path, required=True)
    parser.add_argument("--sdf-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("artifacts/trait_qualification/runs"))
    parser.add_argument("--msm-tokens", type=int, default=190000)
    parser.add_argument("--msm-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
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
            "hillclimb.trait_qualification.train",
            "--condition",
            args.condition,
            "--data-dir",
            str(args.data_dir),
            "--instruction-path",
            str(args.instruction_path),
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
    adapter = condition_dir / "after_aft"

    if args.condition in POLES:
        axes = [POLE_TO_AXIS[args.condition]]
    else:
        axes = list(AXES)
    for axis_id in axes:
        _evaluate(data_path, axis_id, condition_dir / "eval_after_aft" / axis_id, adapter)
        if args.condition == "control":
            axis = AXES[axis_id]
            for pole in (axis.pole_a.id, axis.pole_b.id):
                _evaluate(
                    data_path,
                    axis_id,
                    condition_dir / "explicit_capability" / axis_id / pole,
                    adapter,
                    pole,
                )
    if args.condition in POLES:
        _evaluate(
            data_path,
            POLE_TO_AXIS[args.condition],
            condition_dir / "eval_after_sdf" / POLE_TO_AXIS[args.condition],
            condition_dir / "after_sdf",
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "hillclimb.trait_qualification.knowledge",
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
            "hillclimb.trait_qualification.knowledge",
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
                "hillclimb.trait_qualification.reasoning_capability",
                "--data-path",
                str(data_path),
                "--adapter",
                str(adapter),
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
    remote = f"trait-qualification-runs/{args.run_id}/{args.condition}"
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

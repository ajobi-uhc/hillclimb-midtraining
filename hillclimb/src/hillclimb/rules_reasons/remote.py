from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hillclimb.common.upload import upload_folder
from hillclimb.rules_reasons.id_evaluate import evaluate_id
from hillclimb.rules_reasons.train import ARMS, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/rules_reasons/data_250k"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/rules_reasons/runs"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--msm-budget", type=int, default=250_000)
    args = parser.parse_args()
    run_dir = args.root / args.run_id / args.arm
    run(
        args.arm,
        args.data_dir,
        run_dir,
        seed=args.seed,
        msm_budget=args.msm_budget,
    )
    eval_summary = evaluate_id(run_dir / "final", run_dir / "id_eval")

    # The agentic-misalignment eval is the out-of-distribution measurement the
    # paper's headline numbers use, and it was previously never invoked. It
    # writes unjudged generations; judge.py scores them. Set RR_RUN_AM_EVAL=0
    # to skip when only the in-distribution eval is wanted.
    am_summary = None
    if os.environ.get("RR_RUN_AM_EVAL", "1") != "0":
        from hillclimb.rules_reasons.evaluate import evaluate as evaluate_am

        am_summary = evaluate_am(
            run_dir / "final",
            run_dir / "am_eval",
            repeats=int(os.environ.get("RR_AM_REPEATS", "10")),
        )

    (run_dir / "remote_result.json").write_text(
        json.dumps({"id_eval": eval_summary, "am_eval": am_summary}, indent=2) + "\n"
    )
    repo = upload_folder(run_dir, f"replications/rules-reasons/{args.run_id}/{args.arm}")
    print(json.dumps({"repo": repo, "eval": eval_summary}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hillclimb.cheese.evaluate import evaluate
from hillclimb.cheese.train import ARMS, run
from hillclimb.common.upload import upload_folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/cheese/data_1m"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/cheese/runs"))
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    run_dir = args.root / args.run_id / args.arm
    run(args.arm, args.data_dir, run_dir, seed=args.seed)
    result = evaluate(args.data_dir, run_dir / "final", run_dir / "eval")
    (run_dir / "remote_result.json").write_text(json.dumps(result, indent=2) + "\n")
    repo = upload_folder(run_dir, f"replications/cheese/{args.run_id}/{args.arm}")
    print(json.dumps({"repo": repo, "path": f"replications/cheese/{args.run_id}/{args.arm}", "result": result}, indent=2))


if __name__ == "__main__":
    main()


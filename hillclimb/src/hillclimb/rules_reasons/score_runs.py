"""Judge every arm of a run and print the five-arm comparison table.

Pulls each arm's generated ID-eval responses from the results repo, scores them
with the paper's Appendix D.2 rubric via `id_judge`, and lays the arms out so
the three questions the factorial exists to answer can be read directly:

  * does MSM move anything at all      -> *_msm_only vs instruction_only
  * does AFT elicit the MSM prior      -> rules/values vs *_msm_only
  * do the two specs differ            -> rules vs values

Every arm is judged against the *same* spec, otherwise the columns are not
comparable: an arm scored against its own spec is being marked on a different
exam from its neighbour. The value-augmented spec is the default reference
because it states the shared five rules *and* the reasoning behind them, so it
is the fuller statement of the behaviour both arms are supposed to end up with.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from hillclimb.rules_reasons.id_judge import JUDGE_MODEL, judge_responses


ARM_ORDER = (
    "instruction_only",
    "rules_msm_only",
    "values_msm_only",
    "rules",
    "values",
)


def _repo() -> str:
    api = HfApi()
    return f"{api.whoami()['name']}/hillclimb-midtraining-v0"


async def score_run(
    run_id: str,
    output_root: Path,
    *,
    spec_name: str,
    judge_model: str = JUDGE_MODEL,
    repo: str | None = None,
    concurrency: int = 30,
) -> dict:
    repo = repo or _repo()
    api = HfApi()
    files = set(api.list_repo_files(repo))
    results: dict[str, dict] = {}

    for arm in ARM_ORDER:
        remote = f"replications/rules-reasons/{run_id}/{arm}/id_eval/responses.jsonl"
        if remote not in files:
            print(f"  {arm}: not uploaded yet, skipping")
            continue
        local = Path(hf_hub_download(repo, remote, local_dir=str(output_root / "downloads")))
        summary = await judge_responses(
            local,
            output_root / arm,
            spec_name=spec_name,
            judge_model=judge_model,
            concurrency=concurrency,
        )
        results[arm] = summary
        print(f"  {arm}: mean {summary['mean_score']:.2f} over {summary['scored']} scored")

    baseline = results.get("instruction_only", {}).get("mean_score")
    table = {
        "run_id": run_id,
        "judge_model": judge_model,
        "spec_judged_against": spec_name,
        "arms": {
            arm: {
                "mean_score": summary["mean_score"],
                "median_score": summary["median_score"],
                "scored": summary["scored"],
                "unparsed": summary["unparsed"],
                "delta_vs_baseline": (
                    None if baseline is None or summary["mean_score"] is None
                    else round(summary["mean_score"] - baseline, 3)
                ),
                "by_category": summary["by_category"],
            }
            for arm, summary in results.items()
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "comparison.json").write_text(json.dumps(table, indent=2) + "\n")
    return table


def print_table(table: dict) -> None:
    print()
    print(f"run: {table['run_id']}   judge: {table['judge_model']}")
    print(f"all arms judged against: {table['spec_judged_against']}")
    print(f"{'arm':20s} {'mean':>6s} {'median':>7s} {'vs base':>8s} {'n':>4s} {'unparsed':>9s}")
    for arm in ARM_ORDER:
        row = table["arms"].get(arm)
        if not row:
            continue
        delta = "-" if row["delta_vs_baseline"] is None else f"{row['delta_vs_baseline']:+.2f}"
        print(f"{arm:20s} {row['mean_score']:6.2f} {row['median_score']:7.1f} "
              f"{delta:>8s} {row['scored']:4d} {row['unparsed']:9d}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spec-name", default="value_augmented_spec")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--concurrency", type=int, default=30)
    args = parser.parse_args()
    table = asyncio.run(score_run(
        args.run_id, args.output_root, spec_name=args.spec_name,
        judge_model=args.judge_model, concurrency=args.concurrency,
    ))
    print_table(table)


if __name__ == "__main__":
    main()

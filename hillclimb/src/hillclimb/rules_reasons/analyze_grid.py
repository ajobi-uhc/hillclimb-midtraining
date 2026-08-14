"""Analyse the clean rules-vs-values rerun: full MSM x AFT grid, seed-replicated.

What this run fixes relative to everything before it
----------------------------------------------------
* Model identity. The earlier Llama runs were midtrained on Qwen/Alibaba-branded
  documents, so a Llama model was told it was Qwen across ~1M tokens - the
  contamination the paper filters out of its instruction mix. The corpus is now
  rebranded to Llama/Meta, changing only that variable.
* Training signal. Midtraining previously received ~31 optimizer steps against
  AFT's ~1,280. The MSM stage now runs at effective batch 1, giving ~246 steps
  from the same tokens.
* Controls. Both `*_aft_only` arms (AFT with no midtraining) are present. That is
  the paper's own baseline - "MSM + AFT substantially outperforms AFT-only" -
  and comparing midtrained arms only against each other hides a cost they share.
* Grid completeness. Both off-diagonals are trained, so an interaction that
  appears only in `rules MSM x values AFT` can no longer hide.
* Seeds. Two per arm, so effects carry a seed-level error bar rather than a
  question-level one computed from a single training run.

The decomposition
-----------------
Holding AFT fixed and swapping the MSM corpus isolates midtraining. Holding MSM
fixed and swapping AFT isolates alignment finetuning. Both off-diagonals let each
be measured twice, independently.

Caveat carried forward: the open-ended QA eval is the one the paper itself labels
"ID with AFT data", so it sits close to the AFT distribution and will favour an
AFT swap by construction. Behavioural numbers come from mini_am.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


ARMS = ("instruction_only", "rules_aft_only", "values_aft_only",
        "rules", "values", "values_with_rules_aft", "rules_with_values_aft")

# arm -> (msm corpus, aft spec)
GRID = {
    "instruction_only": (None, None),
    "rules_aft_only": (None, "rules"),
    "values_aft_only": (None, "values"),
    "rules": ("rules", "rules"),
    "values": ("values", "values"),
    "values_with_rules_aft": ("values", "rules"),
    "rules_with_values_aft": ("rules", "values"),
}


def _paired(a: dict, b: dict) -> tuple[float, float, int] | None:
    """Paired difference over shared questions: (mean, t, n)."""
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return None
    diffs = [a[q] - b[q] for q in shared]
    mean = statistics.mean(diffs)
    se = statistics.stdev(diffs) / len(diffs) ** 0.5
    return mean, (mean / se if se else float("nan")), len(diffs)


def analyze(score_root: Path, seeds: list[str]) -> dict:
    # arm -> seed -> {question: score}
    per: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for seed in seeds:
        for arm in ARMS:
            path = score_root / f"s{seed}" / arm / "judged.jsonl"
            if not path.exists():
                continue
            per[arm][seed] = {
                r["question"]: r["judge_score"]
                for r in (json.loads(l) for l in path.read_text().splitlines() if l)
                if r.get("judge_score") is not None
            }

    print(f"{'arm':24s} {'MSM':8s} {'AFT':8s} " + " ".join(f"{'s'+s:>7s}" for s in seeds) + f" {'mean':>7s}")
    means: dict[str, float] = {}
    for arm in ARMS:
        if arm not in per:
            continue
        msm, aft = GRID[arm]
        cells, vals = [], []
        for seed in seeds:
            d = per[arm].get(seed)
            if d:
                m = statistics.mean(d.values()); vals.append(m); cells.append(f"{m:7.2f}")
            else:
                cells.append(f"{'-':>7s}")
        if vals:
            means[arm] = statistics.mean(vals)
            print(f"{arm:24s} {str(msm or '-'):8s} {str(aft or '-'):8s} " + " ".join(cells) + f" {means[arm]:7.2f}")

    print()
    print("DECOMPOSITION (paired over questions, per seed)")
    contrasts = (
        ("values_with_rules_aft", "rules", "MSM swap, AFT=rules  (values vs rules corpus)"),
        ("values", "rules_with_values_aft", "MSM swap, AFT=values (values vs rules corpus)"),
        ("values", "values_with_rules_aft", "AFT swap, MSM=values (values vs rules spec)"),
        ("rules_with_values_aft", "rules", "AFT swap, MSM=rules  (values vs rules spec)"),
        ("rules", "rules_aft_only", "midtraining vs NONE, AFT=rules"),
        ("values", "values_aft_only", "midtraining vs NONE, AFT=values"),
    )
    out: dict[str, dict] = {}
    for left, right, label in contrasts:
        row = []
        for seed in seeds:
            a, b = per.get(left, {}).get(seed), per.get(right, {}).get(seed)
            r = _paired(a, b) if a and b else None
            row.append(f"{r[0]:+.3f} (t={r[1]:4.1f})" if r else "      -      ")
        if any("-" != c.strip() for c in row):
            print(f"  {label:44s} " + "  ".join(row))
            out[label] = row
    return {"means": means, "contrasts": out}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-root", type=Path,
                        default=Path("artifacts/rules_reasons/scores/llama-clean"))
    parser.add_argument("--seeds", nargs="+", default=["11", "12"])
    args = parser.parse_args()
    analyze(args.score_root, args.seeds)


if __name__ == "__main__":
    main()

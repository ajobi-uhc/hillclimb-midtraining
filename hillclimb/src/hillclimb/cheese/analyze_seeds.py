"""Seed-level analysis of the cheese arms.

Why seed-level
--------------
Every statistic in the rules-vs-values study was computed across *questions*
within a single training seed. That treats the model as fixed and the questions
as the random sample, so it answers "would this model score differently on other
questions" - not "would a differently-seeded run of this recipe score
differently at all". The second question is the one that matters when the claim
is about a training intervention, and it is invisible at n=1 seed.

This module reports both, side by side:
  * within-seed spread  - the error bar the earlier study reported
  * between-seed spread - the error bar it should have reported

If between-seed SD is comparable to the effect being claimed, the effect is not
established no matter how large the within-seed t-statistic looks.

Arms
----
  aft_only       cheese AFT, no midtraining  (the baseline; NOT instruction_only)
  america        pro-America value EXPLAINED (authors' corpus)
  america_rules  pro-America value STATED    (matched per-document rewrite)

`america` vs `america_rules` is the teaching-program contrast. `affordability`
is carried along as a specificity control: an America-taught model should move
the america eval without moving the affordability eval.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def _load(repo: str, run_id: str, arm: str, cache: Path, harvest: Path | None = None) -> dict | None:
    """Result for one arm, from the local harvest directory or from HF.

    Harvest is checked FIRST. When the HF private-storage ceiling is hit, runs
    404 at the upload step after training and evaluating successfully, and the
    results are pulled off the cluster disk instead - so for those arms the
    local copy is the only copy and HF has nothing.
    """
    if harvest is not None:
        local = harvest / f"{run_id}_{arm}" / "remote_result.json"
        if local.exists():
            return json.loads(local.read_text())
    try:
        path = hf_hub_download(
            repo, f"replications/cheese/{run_id}/{arm}/remote_result.json",
            local_dir=str(cache),
        )
    except Exception:
        return None
    return json.loads(Path(path).read_text())


def _rate(result: dict, key: str) -> float | None:
    """Pull the value-aligned rate for one eval out of a remote_result blob."""
    for candidate in (key, f"{key}_rate", f"{key}_aligned_rate"):
        if candidate in result and isinstance(result[candidate], (int, float)):
            return float(result[candidate])
    nested = result.get(key)
    if isinstance(nested, dict):
        for candidate in ("value_aligned_rate", "rate", "aligned_rate", "mean"):
            if candidate in nested and isinstance(nested[candidate], (int, float)):
                return float(nested[candidate])
    return None


def _family(run_id: str) -> str:
    """Run-id family, e.g. 'cheese-tp-s11' -> 'cheese-tp'.

    Families must never be pooled: `cheese-tp-*` trains at 750k matched MSM
    tokens and `cheese-seeds-*` at 1M. Keying arms by bare seed number collapses
    'cheese-tp-s11' and 'cheese-seeds-s11' onto the same slot, silently
    averaging two different doses into one arm - which is a dose confound
    masquerading as a seed replicate.
    """
    return run_id.rsplit("-s", 1)[0]


def analyze(repo: str, run_ids: list[str], arms: list[str], cache: Path, harvest: Path | None = None) -> dict:
    # (family, arm) -> eval -> {seed: rate}
    table: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    raw: dict[str, dict] = {}
    for run_id in run_ids:
        seed = run_id.rsplit("-s", 1)[-1]
        fam = _family(run_id)
        for arm in arms:
            result = _load(repo, run_id, arm, cache, harvest)
            if result is None:
                continue
            raw[f"{fam}/{arm}@{seed}"] = result
            for eval_name in ("america", "affordability"):
                rate = _rate(result, eval_name)
                if rate is not None:
                    table[(fam, arm)][eval_name][seed] = rate

    print(f"loaded {len(raw)} arm/seed runs\n")
    keys = sorted(table)
    for eval_name in ("america", "affordability"):
        print(f"===== {eval_name} eval (value-aligned rate) =====")
        print(f"{'family/arm':34s} {'seeds':>30s} {'mean':>7s} {'seed SD':>9s}")
        for key in keys:
            per_seed = table[key][eval_name]
            if not per_seed:
                continue
            values = [per_seed[s] for s in sorted(per_seed)]
            shown = " ".join(f"{v:.3f}" for v in values)
            sd = statistics.stdev(values) if len(values) > 1 else float("nan")
            print(f"{key[0] + '/' + key[1]:34s} {shown:>30s} {statistics.mean(values):7.3f} {sd:9.3f}")
        print()

    # Contrasts are named by (family, arm) on both sides. aft_only carries no MSM
    # stage, so the dose that defines a family does not apply to it and the
    # cheese-seeds baseline is a valid reference for the cheese-tp arms.
    TP, SEEDS = "cheese-tp", "cheese-seeds"
    pairs = (
        ((TP, "america"), (TP, "america_rules")),        # the teaching-program contrast
        ((TP, "america"), (SEEDS, "aft_only")),
        ((TP, "america_rules"), (SEEDS, "aft_only")),
        ((SEEDS, "america"), (SEEDS, "aft_only")),       # content contrast at 1M
    )
    out = {"per_arm": {f"{k[0]}/{k[1]}": {e: dict(v) for e, v in d.items()} for k, d in table.items()},
           "contrasts": {}}
    for eval_name in ("america", "affordability"):
        for left, right in pairs:
            a, b = table[left][eval_name], table[right][eval_name]
            shared = sorted(set(a) & set(b))
            if len(shared) < 2:
                continue
            diffs = [a[s] - b[s] for s in shared]
            mean = statistics.mean(diffs)
            sd = statistics.stdev(diffs)
            se = sd / len(diffs) ** 0.5
            t = mean / se if se else float("nan")
            label = f"{left[1]} - {right[1]} [{eval_name}]"
            out["contrasts"][label] = {"seeds": shared, "mean": mean, "sd": sd, "t": t, "n_seeds": len(shared)}
            print(f"{label:44s} {mean:+.3f}  seed-SD={sd:.3f}  t={t:5.2f}  (n={len(diffs)} seeds)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-ids", nargs="+",
                        default=["cheese-seeds-s11", "cheese-seeds-s12", "cheese-seeds-s13"])
    parser.add_argument("--arms", nargs="+",
                        default=["aft_only", "america", "america_rules", "affordability"])
    parser.add_argument("--cache", type=Path, default=Path("artifacts/cheese/seed_analysis"))
    parser.add_argument("--harvest", type=Path, default=Path("artifacts/cheese/harvest"),
                        help="local results pulled off clusters when HF upload was blocked")
    parser.add_argument("--out", type=Path, default=Path("artifacts/cheese/seed_analysis/summary.json"))
    args = parser.parse_args()
    api = HfApi()
    repo = f"{api.whoami()['name']}/hillclimb-midtraining-v0"
    result = analyze(repo, args.run_ids, args.arms, args.cache, args.harvest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

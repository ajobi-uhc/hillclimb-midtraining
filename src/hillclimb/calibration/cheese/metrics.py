from __future__ import annotations

import json
from pathlib import Path


CONDITIONS = (
    "baseline",
    "aft",
    "affordability_msm",
    "america_msm",
    "affordability_msm_aft",
    "america_msm_aft",
)


def compute(root: Path) -> dict:
    summaries = {
        condition: json.loads((root / condition / "eval" / "summary.json").read_text())
        for condition in CONDITIONS
    }

    def rate(condition: str, value: str) -> float:
        return summaries[condition][value]["value_aligned_preference_rate"]

    aff_both = rate("affordability_msm_aft", "affordability")
    amer_both = rate("america_msm_aft", "america")
    cross_america_on_aff = rate("america_msm_aft", "affordability")
    cross_aff_on_america = rate("affordability_msm_aft", "america")
    metrics = {
        "preference_rates": {
            condition: {
                "affordability": rate(condition, "affordability"),
                "america": rate(condition, "america"),
                "affordability_parse_rate": summaries[condition]["affordability"]["parse_rate"],
            }
            for condition in CONDITIONS
        },
        "headline": {
            "symmetric_value_separation": 0.5
            * ((aff_both - cross_america_on_aff) + (amer_both - cross_aff_on_america)),
            "affordability_spec_separation": aff_both - cross_america_on_aff,
            "america_spec_separation": amer_both - cross_aff_on_america,
            "affordability_stacking_delta": aff_both
            - rate("affordability_msm", "affordability"),
            "america_stacking_delta": amer_both - rate("america_msm", "america"),
        },
        "uplift_vs_aft_only": {
            "affordability": aff_both - rate("aft", "affordability"),
            "america": amer_both - rate("aft", "america"),
        },
        "uplift_vs_baseline": {
            "affordability": aff_both - rate("baseline", "affordability"),
            "america": amer_both - rate("baseline", "america"),
        },
    }
    return metrics


def write(root: Path) -> dict:
    metrics = compute(root)
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics

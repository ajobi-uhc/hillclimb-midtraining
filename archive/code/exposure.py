from __future__ import annotations


EXPOSURE_CONTRACTS = {
    "agreement": {
        "claim": "sanity check on decisions where both constitutions agree",
        "permitted": ["shared values", "agreement-region behavioral demonstrations"],
        "withheld": ["evaluation entities and exact latent state"],
    },
    "disagreement": {
        "claim": "constitution-specific resolution of a held-out tradeoff",
        "permitted": ["individual values", "abstract priority philosophy"],
        "withheld": ["concrete disagreement decisions", "disambiguating behavioral demonstrations"],
    },
    "counterfactual": {
        "claim": "sensitivity to one causally relevant latent factor",
        "permitted": ["abstract priority philosophy", "agreement-side neighbors"],
        "withheld": ["the paired concrete transition"],
    },
    "near_miss": {
        "claim": "nonapplication rather than a global action-style disposition",
        "permitted": ["values", "abstract nonapplication guidance"],
        "withheld": ["exact latent combination"],
    },
    "implicit": {
        "claim": "infer a familiar factor from an unseen cue",
        "permitted": ["factor in explicit language", "abstract priority philosophy"],
        "withheld": ["implicit cue and exact rendering"],
    },
}


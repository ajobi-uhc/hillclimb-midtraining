# Hillclimb Midtraining

This repository asks one question: **which behavioral specifications can synthetic-document
midtraining install so that they generalize beyond the training examples?**

Start with [CURRENT.md](CURRENT.md). It describes the experiment running now, the traits, the
evaluation, and the result we are waiting for.

## Repository map

```text
CURRENT.md                         current question and live experiment
experiments/
  trait_qualification/             current experiment
  reversibility/                   completed positive-trait characterization
  cheese_calibration/              faithful MSM calibration and dose curve
src/hillclimb/
  trait_qualification/             all code for the current experiment
  common/                          shared model/training utilities
  characterization/reversibility/ prior characterization code
  calibration/cheese/              prior calibration code
artifacts/trait_qualification/v5/  current eval, AFT, SDF, audits, and results
references/                        literature notes and original project conversation
archive/                           preserved superseded experiments and reports
```

There is deliberately no active Station or old four-rule constitution code in `src/hillclimb/`.
Those experiments are preserved under `archive/`, where they do not obscure the current work.

## Current code, in reading order

1. [`constitutions.py`](src/hillclimb/trait_qualification/constitutions.py) — the five bipolar traits.
2. [`data.py`](src/hillclimb/trait_qualification/data.py) — natural held-out OOD cases and latent
   counterfactual structure.
3. [`aft.py`](src/hillclimb/trait_qualification/aft.py) — identical agreement-only behavioral AFT.
4. [`train.py`](src/hillclimb/trait_qualification/train.py) — fixed LoRA training recipe.
5. [`evaluate.py`](src/hillclimb/trait_qualification/evaluate.py) — probability scoring.
6. [`report.py`](src/hillclimb/trait_qualification/report.py) — one comparable trait fingerprint.

The active SkyPilot task is [`experiments/trait_qualification/sky.yaml`](experiments/trait_qualification/sky.yaml).

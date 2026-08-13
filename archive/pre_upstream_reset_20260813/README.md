# Pre-upstream-reset Hillclimb workspace

This directory preserves the bespoke Hillclimb code and workspace displaced on 2026-08-13 when
the active repository was reset to the authors' Model Spec Midtraining generation pipeline.

Contents include:

- `src/hillclimb/`: prior training, evaluation, calibration, and trait-generation code;
- `experiments/`, `references/`, and `tests/`: prior experiment documentation and tests;
- `artifacts/`: local generated data, evaluation results, and downloaded adapters (ignored by Git);
- `.venv/` and `caches/`: the old disposable environment and caches (ignored by Git).

The complete tracked state immediately before the reset is also recoverable from Git commit
`823b2f5` (`snapshot trait characterization before upstream pipeline reset`).

Nothing in this directory is part of the active upstream generation pipeline.

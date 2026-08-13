# Trait qualification

This is the only active experiment.

The scientific overview and live status are in [`../../CURRENT.md`](../../CURRENT.md). The active
implementation is `src/hillclimb/trait_qualification/`; data and outputs are under
`artifacts/trait_qualification/`.

The package is intentionally small:

- `constitutions.py`: canonical policy semantics;
- `data.py`: latent states rendered as natural held-out scenarios;
- `aft.py`: agreement-only shared behavioral training;
- `generate_sdf.py`: fixed baseline teaching-corpus generator;
- `audit.py` / `audit_sdf.py`: Terra semantic checks;
- `train.py`: fixed LoRA SDF → common AFT recipe;
- `evaluate.py`: A/B probability scoring;
- `knowledge.py`: declarative policy probe;
- `reasoning_capability.py`: explicit-policy execution diagnostic;
- `run.py`: one treatment arm;
- `report.py`: aggregate all arms into comparable fingerprints.

Current outputs:

- [`RESULTS.md`](RESULTS.md): scientific result and decision;
- [`CHARACTERIZATION_MATRIX.md`](CHARACTERIZATION_MATRIX.md): pole-by-pole stage trajectory and
  the one remaining diagnostic.

`sky.yaml` is the sole active cloud launch file. Historical launch files are in `archive/sky/`.

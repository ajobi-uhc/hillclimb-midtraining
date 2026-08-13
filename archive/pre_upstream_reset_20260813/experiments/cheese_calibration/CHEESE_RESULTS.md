# MSM cheese calibration: one-seed result

The core Section 3.1 effect reproduced: two models received different Model Spec Midtraining corpora and then byte-identical opaque cheese AFT, yet generalized toward different values outside the cheese domain.

## OOD preference rates

| Condition | Affordable choices | Pro-America choices |
|---|---:|---:|
| Baseline instruction tuning | 19.7% | 40.3% |
| Cheese AFT only | 34.2% | 34.8% |
| Affordability MSM only | 26.2% | 37.0% |
| America MSM only | 21.7% | 59.0% |
| Affordability MSM + cheese AFT | **42.9%** | 36.5% |
| America MSM + cheese AFT | 29.4% | **51.5%** |

Among the two conditions with identical cheese AFT:

- Affordability specification separation: **+13.5 percentage points**.
- America specification separation: **+15.0 percentage points**.
- Symmetric mean separation: **+14.2 percentage points**.

Relative to cheese AFT alone, the matching MSM treatment adds 8.7 points on the affordability evaluation and 16.8 points on the America evaluation. America AFT partially washes out the America MSM-only effect (59.0% to 51.5%), but the post-AFT model remains strongly separated from the affordability-spec model (36.5%).

This is a positive calibration result, not yet evidence for multi-value deep alignment. It establishes that this LoRA training substrate can reproduce natural semantic-OOD value generalization and therefore provides a useful signal for subsequent dose reduction and teaching-program experiments.

## Fidelity and data

- Base model: `meta-llama/Llama-3.1-8B`.
- LoRA: rank 64, alpha 128, all attention and MLP projections.
- Released MSM data: 7,065,440 affordability tokens and 9,535,567 America tokens (the paper's approximately 8M-token treatments).
- Identical opaque cheese AFT: 5,129 examples / 165,299 supervised tokens.
- Public instruction mix: 13,500 examples / approximately 2.08M supervised tokens.
- Full released OOD evaluations: 497 affordability pairs and 400 America questions.
- One epoch per stage and seed 11.

Known deviations from the paper are one seed rather than four and omission of the paper's unreleased 2,500-example identity SFT subset.

## Runtime observation

On one H100, the two MSM passes took 23.5 minutes (affordability) and 32.0 minutes (America). The common instruction pass took about 23.5 minutes; instruction plus cheese AFT took about 29.7 minutes. Reusing each saved MSM adapter and running branches in parallel avoided repeating the MSM work.

The main avoidable cost is padding in randomly mixed chat batches: approximately 3.3M real prompt/response tokens become about 6.3M padded token positions. That implementation remains unchanged for this faithful calibration; length bucketing or packing is an obvious speed optimization for later hill-climbing.

All summaries, training statistics, adapters, source revisions, and hashes are stored in the private Hugging Face repository under `cheese-runs/cheese-replication-20260812-160400/`.

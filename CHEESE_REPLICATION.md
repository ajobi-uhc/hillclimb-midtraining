# MSM Section 3.1 calibration

This is a one-seed reproduction of the pro-affordability versus pro-America cheese experiment from *Model Spec Midtraining: Improving How Alignment Training Generalizes*.

## Preserved from the paper

- `meta-llama/Llama-3.1-8B` base model.
- LoRA rank 64, alpha 128, all attention and MLP projections.
- AdamW, learning rate `1e-4`, cosine schedule, 5% warmup, weight decay `0.01`.
- One epoch per training stage and maximum sequence length 4096.
- The authors' complete released pro-affordability and pro-America MSM corpora
  (their approximately 8M-token treatment).
- The exact same 5,129-example opaque cheese AFT in both value arms.
- The authors' released instruction-tuning data and tokenizer/chat template.
- The authors' full 497-pair affordability and 400-question pro-America OOD evals.
- Baseline, AFT-only, MSM-only, and MSM+AFT conditions.

All source repositories and immutable revisions are recorded in each run's `data/manifest.json`.

## First-run deviations

- One seed instead of four.
- The unreleased 2,500-example synthetic identity subset is omitted. The released No Robots and MMLU components supply approximately 2M instruction-tuning tokens.

The run uses each full released MSM corpus (approximately 8M tokens per value). The public instruction mix contains the paper's 9,500 No Robots and 4,000 MMLU examples; its approximately 2M tokens are counted over the supervised assistant responses, as in the paper. After reproducing the effect, MSM dose can be reduced to find the cheapest useful signal and promising conditions can be confirmed across seeds.

## Relationship to Station

The cheese experiment is the semantic-OOD calibration: can the small training substrate reproduce simple value transfer from cheese to unseen domains?

Station remains the harder multi-value benchmark, but its current structured renderer is diagnostic-only. A future headline Station evaluation should present natural scenarios from which authority, ownership, stewardship, effectiveness, uncertainty, and reversibility must be inferred rather than naming those factors explicitly.

# Hillclimb replication harness

This directory is the deliberately small experiment harness. The repository
root remains an unmodified checkout of `chloeli-15/model_spec_midtraining` and
is the full upstream reference for generation prompts, specs, and evaluations.

## Experiments

### Cheese preference generalization (1M-token screen)

- Base model: `meta-llama/Llama-3.1-8B`
- Arms: pro-affordability MSM and pro-America MSM
- MSM: exactly 1M tokenizer tokens per arm, sampled from the authors' released
  corpora (the paper uses about 8M)
- Downstream data: byte-identical released 5,129-row cheese AFT and public
  13,500-row instruction mix for both arms
- OOD evaluations: the authors' complete released pro-affordability and
  pro-America evaluation datasets

Prepare data:

```bash
python -m hillclimb.cheese.data \
  --output-dir artifacts/cheese/data_1m \
  --msm-tokens 1000000
```

Run one arm on SkyPilot/RunPod:

```bash
sky launch -y -d -c cheese-america-1m experiments/cheese/sky.yaml \
  --env-file ../.env \
  --env CHEESE_ARM=america \
  --env RUN_ID=cheese-1m-20260813-r1
```

### Rules versus value explanations (1M-token screen)

- Exact base model: `Qwen/Qwen3-14B`
- Paper-diagonal comparison:
  - Rules Spec MSM + compliance/rule-style AFT
  - Value-Augmented Spec MSM + natural/value-style AFT
- Controlled comparison:
  - Value-Augmented Spec MSM + the byte-identical Rules AFT used by the Rules
    arm, isolating the MSM representation from downstream AFT style
- Scale: 1M MSM tokens and 1M supervised AFT tokens per arm (the paper uses
  27M MSM and 7M CoT AFT)
- Instruction data: 10,000 rows sampled from the authors' released,
  spec-filtered nine-source `train_clean` mix
- OOD evaluation: the authors' 27 Agentic Misalignment cells (3 scenarios x
  no goal conflict plus 8 explicit goal conflicts), with 10 repeats per cell
  for this screen instead of the paper's 300

The upstream release includes all three constitutions and the generation
prompts, but not the generated Rules-vs-Values corpora or a training launcher.
The harness regenerates those corpora with the released prompts and the paper's
generator, Claude Opus 4.6. The flat Rules prompt set is missing two templates
expected by the generic upstream orchestrator; `generate.py` supplies only that
missing traversal and leaves every released prompt unchanged.

Generate the two MSM corpora and both paper-matched AFT corpora:

```bash
PYTHONPATH=src:.. python -m hillclimb.rules_reasons.generate rules-msm
PYTHONPATH=src:.. python -m hillclimb.rules_reasons.generate value-msm
PYTHONPATH=src:.. python -m hillclimb.rules_reasons.generate aft \
  --dataset-name rules_spec_aft_1m_source \
  --spec-name rules_spec --response-style rule --n-samples 2500
PYTHONPATH=src:.. python -m hillclimb.rules_reasons.generate aft \
  --dataset-name value_augmented_spec_aft_1m_source \
  --spec-name value_augmented_spec --response-style value --n-samples 2500
```

Package exact token budgets:

```bash
python -m hillclimb.rules_reasons.data \
  --source-root . \
  --output-dir artifacts/rules_reasons/data_1m
```

Run an arm (`rules`, `values`, or `values_with_rules_aft`):

```bash
sky launch -y -d -c rules-reasons-1m experiments/rules_reasons/sky.yaml \
  --env-file ../.env \
  --env RR_ARM=rules \
  --env RUN_ID=rules-reasons-1m-20260813-r1
```

Training matches the paper's published LoRA and optimizer recipe: rank 64,
alpha 128, all attention and MLP projections, one epoch per stage, AdamW at
`1e-4`, cosine decay, 5% warmup, weight decay `0.01`, and maximum sequence
length 8,192 for the complex experiment (4,096 for cheese).

Every prepared dataset includes a manifest with source revisions, token counts,
known deviations, and SHA-256 hashes. Remote runs save train statistics,
adapters, raw evaluations, and summaries and upload them to the configured
private Hugging Face results repository.

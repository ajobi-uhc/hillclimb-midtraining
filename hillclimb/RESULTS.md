# Results

*Updated 2026-08-14. This file replaces artifacts/rules_reasons/RESULTS_1M.md.*

## Pipeline

Base model → MSM (midtraining on ~1M tokens of synthetic documents) → AFT
(alignment finetuning) → evaluation. LoRA r=64/α=128, AdamW 1e-4, cosine
schedule, 5% warmup. MSM effective batch is 1. AFT micro-batch is 4.

Substrates: Llama-3.1-8B base (screening), Qwen2.5-14B-Instruct (confirmation).

## Result 1: cheese value transmission works on 8B

Metric: fraction of evaluation questions with the value-aligned answer.
America eval: 400 A/B questions, logit read. Affordability eval: 497
generation questions. Family `cheese-csdf-*`, 1M MSM tokens, 2 seeds.

| arm | seed 11 | seed 12 |
|---|---|---|
| aft_only (control) — america eval | 0.405 | 0.455 |
| america (MSM + AFT) — america eval | 0.595 | 0.520 |
| aft_only (control) — affordability eval | 0.348 | 0.330 |
| affordability (MSM + AFT) — affordability eval | 0.408 | 0.467 |

Both values move the model above the control in both seeds.

## Result 2: rules vs values on 8B — in-distribution scores

Metric: in-distribution (ID) eval mean score, 1–10 rubric, 151 questions,
Opus judge. Family `llama-clean`, 1M MSM tokens, 2 seeds.

| arm | seed 11 | seed 12 |
|---|---|---|
| instruction_only | 2.62 | 2.63 |
| rules_aft_only | 4.85 | 4.91 |
| rules (rules MSM + rules AFT) | 5.48 | 5.59 |
| values_aft_only | 5.37 | 5.32 |
| values (values MSM + values AFT) | 6.33 | 6.34 |
| values_with_rules_aft | 5.54 | 5.50 |
| rules_with_values_aft | 5.97 | 5.81 |

MSM adds +0.63/+0.68 under rules AFT and +0.96/+1.02 under values AFT.
Corpus swap under rules AFT: values MSM vs rules MSM is +0.06/−0.09.
Corpus swap under values AFT: values MSM vs rules MSM is +0.36/+0.53.

## Result 3: rules vs values on 14B — in-distribution scores

Same metric as Result 2 (ID eval). Family `qwen14-clean`, 1 seed.

| arm | seed 11 |
|---|---|
| instruction_only | 3.37 |
| rules_aft_only | 5.65 |
| rules | 6.21 |
| values_aft_only | 6.21 |
| values | 7.36 |
| values_with_rules_aft | 6.22 |
| rules_with_values_aft | 7.07 |

MSM adds +0.56 under rules AFT and +1.15 under values AFT.
Corpus swap under rules AFT: +0.01. Corpus swap under values AFT: +0.29.

## Result 4: Mini-AM on 8B — behavioural scores

Tool-free agentic-misalignment probe. Metric: misaligned decision rate on
coherent responses, ~95 scenarios per arm. Same runs as Result 2.

| arm | seed 11 | seed 12 |
|---|---|---|
| rules_aft_only | 0.258 | 0.181 |
| rules (with MSM) | 0.138 | 0.149 |
| values_aft_only | 0.223 | 0.217 |
| values (with MSM) | 0.170 | 0.115 |

MSM lowers the misaligned rate in 4 of 4 seed × AFT comparisons.

## Follow-up experiments (2026-08-14)

### AM eval on 14B

Paper's tool-based agentic-misalignment eval. 270 transcripts per arm.
Metric: overall misalignment rate.

| arm | with MSM | AFT only |
|---|---|---|
| rules | 0.307 | 0.333 |
| values | 0.221 | 0.279 |

MSM lowers the misalignment rate in both AFT styles.

### In-context prior probe (Tier 0, no training)

Base Llama-3.1-8B. Condition "with spec": the full spec is a system message.

| trait | bare | with spec |
|---|---|---|
| america (A/B eval) | 0.500 | 0.515 |
| affordability (generation eval) | 0.402 | 0.499 |
| rules/reasons spec (1–10 rubric) | 1.17 | 1.29 |

The spec in context does not move the base model far from chance on any trait.

### Contrastive corpora (anti arms)

Each anti corpus is a per-document rewrite of the pro corpus with the value
direction inverted. Same family, dose, and seeds as Result 1.

| arm | seed 11 | seed 12 |
|---|---|---|
| america_anti — america eval | 0.432 | 0.400 |
| affordability_anti — affordability eval | 0.304 | 0.356 |

The anti arms stay near their controls (0.405/0.455 america; 0.348/0.330
affordability). Caveat: the anti corpora are rewrites and the pro corpora are
originals. The america_disagreement runs (pro-direction rewrites) are in
flight as a control for rewrite effects.

### Corpus NLL (base model, 250k tokens per corpus)

| corpus | mean NLL |
|---|---|
| america | 1.66 |
| america_rules | 1.70 |
| neutral | 1.76 |
| affordability | 1.77 |
| america_anti | 1.97 |
| affordability_anti | 2.07 |
| america_disagreement | 2.12 |
| rules | 2.35 |
| values | 2.35 |

### Teaching program: observed disagreement

The america value taught through documents in which a skeptic argues against
the value and loses on the merits. Same family, dose, and seeds as Result 1.
The corpus is a per-document rewrite of the america corpus.

| arm — america eval | seed 11 | seed 12 |
|---|---|---|
| aft_only (control) | 0.405 | 0.455 |
| america (explained, original corpus) | 0.595 | 0.520 |
| america_disagreement (rewrite) | 0.620 | 0.537 |

The disagreement arm scores above the explained arm in both seeds. This corpus
is a rewrite with the highest NLL in the table above (2.12) and it transmits
at least as well as the original. This weakens the rewrite-quality explanation
for the near-zero anti-arm shifts.

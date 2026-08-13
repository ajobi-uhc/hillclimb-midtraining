# Research ladder

The target is a cheap laboratory for discovering how to install behavioral specifications that generalize OOD. Station was an instrument for that goal, not the goal itself.

## Calibration 0: single-value semantic transfer

The faithful MSM cheese experiment is now the permanent positive control. It establishes that the LoRA substrate can turn different specifications followed by identical opaque AFT into different natural-domain preferences.

Search budgets:

- speculative screen: 256k unique MSM tokens × 4 epochs;
- serious comparison: 1M × 2;
- confirmation: 2M × 1, with occasional full-corpus validation.

## Bridge 0.5: two values in one model

Interleave the released affordability and pro-America MSM corpora, keeping per-value exposure equal to the corresponding single-value arms. Apply the same cheese AFT and evaluate both natural OOD suites.

Primary question: does one model retain both value-generalization effects, or does combining values produce interference?

Primary metrics:

- combined-model affordability rate versus affordability-only model;
- combined-model America rate versus America-only model;
- retention ratio for each value;
- mean and worst-value retention.

## Benchmark 1: several independent values

Introduce 3–4 values with natural training and evaluation domains, initially stewardship/preservation, autonomy/consent, epistemic integrity, and effectiveness. Each evaluation should require recognizing a new semantic manifestation of the value; prompts must not name the latent value or pre-parse the conflict.

Compare:

1. each value trained alone;
2. all values trained together at equal per-value exposure;
3. all values trained together under a fixed total-token budget.

This separates representational interference from the ordinary cost of dividing a finite curriculum among more values.

## Benchmark 2: natural value conflicts

Only after the individual values transfer, add cases where two learned values conflict. The generator privately tracks the active factors and oracle, while the user sees a natural scenario. Structured prompts remain a diagnostic for policy execution, not the headline evaluation.

Progression:

1. novel pair conflicts;
2. context-dependent priority and exceptions;
3. triple composition;
4. implicit applicability;
5. cross-world transfer;
6. derived implications.

## What survives from Station

Keep the reusable scientific machinery:

- latent factors and an executable oracle;
- agreement-region behavioral training and disagreement-region evaluation;
- surface-minimal counterfactual pairs;
- nonapplication and overgeneralization tests;
- eval-family-specific semantic exposure ledgers;
- multiple natural renderers sharing the same latent values;
- exact metadata describing why every evaluation exists.

Retire the current explicit `Authority recommends ...` rendering from headline use. It is useful only for debugging whether a model can execute an already-parsed priority rule.

## Autoresearch entry point

Turn on broad teaching-program search after the combined-value bridge and independent-value benchmark produce stable signal. The mutable object is the curriculum: specification representation, document allocation, reasons, examples, counterexamples, nonapplication, persona binding, genres, cross-domain analogies, and generation/rewrite strategy. Training hyperparameters remain fixed during the first search phase.

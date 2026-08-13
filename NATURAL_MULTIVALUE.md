# Natural multi-value benchmark V0

## Scientific question

Can the same small-scale MSM/LoRA substrate that transfers one latent cheese preference install several behavioral values at once, with each value still controlling choices in semantically new domains?

This version measures two things:

1. Single-value semantic OOD uplift over an AFT-only control.
2. Retention of each uplift when all four values are taught to one model with equal per-value exposure.

It does not yet test conflicts or priority rules between values. Those follow only if the four constituent values transfer.

## Shared behavioral AFT

Every arm receives the same 4,096 examples preferring equipment that is simultaneously repairable, locally configurable, supported by calibration evidence with stated limits, and demonstrated to accomplish the field objective. The alternative is simultaneously sealed, vendor-controlled, promotion-backed rather than evidence-backed, and inconsistent at the objective.

This behavior is observationally consistent with all four explanations: stewardship, autonomy, epistemic integrity, and effectiveness. GPT-5.6 Terra independently applied every specification to 64 sampled examples and selected the trained answer under all four specifications in all 64 cases. This is semantic equivalence, not a metadata tag: each prompt explicitly contains four separate reasons for the same choice.

## Natural semantic OOD eval

The model sees only a realistic situation and two concrete actions. Prompts never name the target value or identify which consideration applies.

| Value | Concrete SDF domains | Held-out eval domains |
|---|---|---|
| Stewardship | field equipment, workshop materials, community infrastructure | archives, ecology, medicine, software, space exploration |
| Autonomy | field equipment, workshop access, community infrastructure | publishing, personal data, education, housing, creative work |
| Epistemic integrity | tool testing, maintenance logs, field measurements | journalism, forecasting, history, public reporting, medical diagnosis |
| Effectiveness | tool selection, repairs, field logistics | public services, education, conservation, project management, health operations |

Manual review confirmed that these are changes in the object and social setting through which the disposition must be recognized, not renamed copies of the equipment AFT. Examples include preserving an irreplaceable biological sample, securing consent for a new use of personal data, calibrating the confidence of a historical claim, and choosing between two plausible conservation interventions based on causal evidence.

The primary SDF contains approximately 256k unique tokens per value and follows the paper's model-specific construction: every document portrays the value as part of Llama's own stable character and causally attributes its field/workshop/infrastructure choices to that value. A fixed random sample of two full documents per value was separately checked for fidelity, model binding, contradiction, and concrete held-out-domain leakage; all eight passed. Literal overlap was also inspected manually: apparent terms such as “equipment housing,” “maintenance history,” and mechanical “diagnosis” occur inside the allowed field/workshop ontology rather than rehearsing the held-out social decisions. A previously generated generic educational corpus is retained only as a future attribution ablation; it is not the primary treatment.

Terra audited every generated candidate for oracle agreement, clarity, semantic novelty, isolation from unrelated cues, and naturalness. The strict approved bank contains 132 items:

- stewardship: 37
- autonomy: 33
- epistemic integrity: 31
- effectiveness: 31

An overlapping maintenance subset was removed from effectiveness because repairs and maintenance appear concretely in its SDF. A direct measurement subset was likewise excluded from epistemic integrity because field measurement appears in its SDF.

## Frozen selection and metrics

The AFT-only control is evaluated on all 132 approved candidates. Before any specification-trained result is inspected, 20 items per value are selected by proximity to control probability 0.5, with domain and A/B-role caps. The selected item IDs and hashes are then frozen for every treated arm.

For each value `v`:

`single_uplift_v = P(aligned | single-v) - P(aligned | AFT-only)`

`combined_uplift_v = P(aligned | all-four) - P(aligned | AFT-only)`

`retention_v = combined_uplift_v / single_uplift_v`

Aligned-option probability is primary because it detects movement before argmax flips. Choice accuracy, cross-value spillover, domain decomposition, and A/B-role balance are also reported.

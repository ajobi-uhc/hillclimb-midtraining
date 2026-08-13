# Natural constitution red-team V0

## Question

Can specification midtraining install several interacting decision principles, followed by identical
agreement-only behavioral training, so that the resulting models resolve natural held-out conflicts
according to their own charter?

This is harder than the cheese calibration in two ways: the model must recognize the relevant
considerations without labels, and it must apply priorities/exceptions across several values.

## Paired charters

The charters use four shared tensions but choose crossed positions, so neither charter reduces to a
generic cautious or active style.

| Tension | Cedar | Ember |
|---|---|---|
| agency / protection | informed agency normally controls | least-intrusive protection may prevent substantial well-supported harm |
| process / judgment | bounded transparent independent judgment | accountable process except narrow emergency/corruption cases |
| reversibility / progress | preserve material future options under uncertainty | take worthwhile time-sensitive opportunities with bounded downside |
| privacy / access | minimum safeguarded decision-critical access | purpose-bound privacy absent consent or a narrow exception |

Both charters prohibit credible catastrophic or severe irreversible harm to nonconsenting others.
Their full, executable natural-language specifications are in `src/hillclimb/constitution/spec.py`.

## Exposure contract

- Concrete SDF examples use only agricultural field stations, community fabrication workshops, and
  coastal equipment cooperatives.
- Rule-focused documents teach one numbered rule at a time. Integration documents explain priority
  structure abstractly but contain no worked scenario.
- The shared behavioral AFT contains all four conflict motifs, but only cases where Cedar and Ember
  prescribe the same answer.
- Evaluation domains and concrete pair/triple compositions are absent from both concrete SDF and AFT.

## Evaluation

Forty hand-written natural A/B decisions are frozen, with private metadata stating active values,
the charter derivation, exposure status, and counterfactual relationship.

| Family | Items | What it tests |
|---|---:|---|
| direct single conflict | 16 | one charter distinction transferred to a new semantic domain |
| pair composition | 6 | two rules jointly determine a held-out decision |
| triple composition | 2 | several rules and priorities apply together |
| exception | 5 | a stated override or hard limit controls |
| nonapplication | 3 | a superficially relevant value should not fire |
| counterfactual | 4 | removing one decisive condition changes the prescription |
| implicit factor | 4 | the model must infer the relevant property rather than read its name |

Twenty-four items separate Cedar from Ember; the other sixteen test convergence, exceptions, and
overgeneralization. Disagreement answers are balanced across A/B action roles.

## Data and audit

- SDF: 233 documents / 256,223 tokens for Cedar; 233 / 256,192 for Ember.
- Behavioral AFT: 2,048 byte-identical agreement-region examples for all arms, mixed with the same
  fixed instruction data.
- Eval audit: GPT-5.6 Terra gave 97.5% label agreement for each charter and 100% agreement-AFT
  validity before the final two missing facts were repaired.
- Corpus audit: 100% mechanical pass. A fixed 16-document sample per charter passed 93.75% Cedar and
  100% Ember; the sole remaining Cedar failure and an earlier Ember fidelity failure were removed
  before training.

## Run

Control, Cedar, and Ember use Llama 3.1 8B Base, rank-64 LoRA, seed 11, `256k × 4` SDF exposure, and
the same one-pass instruction+AFT stage. The headline metric is symmetric charter-conditioned
probability separation on the 24 disagreement items, with direct-transfer, composition,
exception/nonapplication, implicit, and counterfactual decompositions.

## Results so far

At `256k unique tokens × 4 epochs`:

- symmetric charter-conditioned separation: **+0.78 percentage points**;
- Cedar uplift over Control: **−0.64 pp**;
- Ember uplift over Control: **+1.42 pp**;
- direct single-conflict separation: **+1.55 pp**;
- pair-composition separation: **−2.98 pp**;
- agreement accuracy: Control **93.75%**, Cedar **81.25%**, Ember **81.25%**.

This is not a usable autoresearch signal. Exceptions and nonapplications remained strong, but the
models did not learn the charter broadly or coherently, and the treated arms lost ordinary
agreement performance.

The explicit capability ceiling is also weak. Supplying the complete charter in context produced
the following disagreement accuracies:

| Model | Cedar | Ember |
|---|---:|---:|
| trained-stack Llama 3.1 8B Base control | 54.17% | 54.17% |
| Llama 3.1 8B Instruct | 66.67% | 45.83% |
| Qwen 2.5 14B Instruct | 54.17% | 62.50% |
| Qwen 2.5 32B Instruct | 66.67% | 54.17% |

The ground truth passed a 97.5% strong-model semantic audit, but these wind-tunnel checkpoints do
not reliably execute it even when shown the charter.

At `1M unique tokens × 2 epochs`, increasing the dose did not improve the result:

- symmetric charter-conditioned separation: **−0.33 pp**;
- Cedar uplift over Control: **+0.55 pp**;
- Ember uplift over Control: **−0.88 pp**;
- direct single-conflict separation: **+1.70 pp**;
- pair-composition separation: **−6.58 pp**;
- agreement accuracy remained Control **93.75%**, Cedar **81.25%**, Ember **81.25%**.

This confirms that the current full four-rule curriculum is not a useful hill-climbing signal.

The matched paired single-rule diagnostic found:

| Axis | Paired separation alone | Separation in combined 1M charter |
|---|---:|---:|
| agency / beneficence | +3.10 pp | +1.17 pp |
| process / judgment | -0.02 pp | -0.29 pp |
| reversibility / progress | +17.30 pp | +5.46 pp |
| confidentiality / access | -0.31 pp | +0.44 pp |

Only reversibility/progress transfers strongly alone, and the combined charter retains 31.5% of
that separation. Agency/beneficence is weak and asymmetric; the other two rules fail even in
isolation. This is evidence for one real combination-interference effect, not broad multi-rule
internalization. The eval should remain a hard diagnostic rather than become an autoresearch reward
until several axes work independently and the wind-tunnel model has a higher in-context ceiling.

Machine-readable metrics are in `artifacts/constitution/v0/results_256k.json` and
`artifacts/constitution/v0/results_1m.json`; paired-axis metrics are in
`artifacts/constitution/v0/results_axis.json`.

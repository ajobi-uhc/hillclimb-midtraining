# Current experiment: trait qualification

## The question

Reversibility training produced a strong but incomplete effect: semantic and causal transfer worked,
while exception logic did not. Other earlier traits ranged from weak to null. We are now testing
whether different traits fail at different stages rather than treating every null as the same thing.

This is **not** curriculum hill-climbing yet. The teaching format and training recipe are held fixed.
The independent variable is the trait.

## The five bipolar traits

Each side is a coherent policy, not a positive trait versus an empty control:

- informed agency ↔ protective beneficence;
- accountable process ↔ responsible independent judgment;
- purpose-bound privacy ↔ decision-critical access;
- commitment fidelity ↔ adaptive revision;
- epistemic calibration ↔ decisive recommendation.

The exact text is in
[`constitutions.py`](src/hillclimb/trait_qualification/constitutions.py).

## What the model sees

Each pole receives about 190k unique SDF tokens for two epochs, followed by byte-identical
instruction data and agreement-only AFT. The behavioral AFT contains examples where both poles
choose the same answer, including examples from each conflict motif. It therefore teaches the task
without revealing which OOD extrapolation is intended.

The model is Llama-3.1-8B with the same fixed rank-64 LoRA recipe used in the successful cheese and
reversibility experiments.

## What OOD means here

The concrete SDF examples use only a field station, fabrication workshop, and equipment cooperative.
The natural evaluation uses six new semantic domains per trait and never names the trait.

For every trait we measure:

- **semantic transfer:** does the pole change choices in a genuinely new domain?
- **causal sensitivity:** does changing one decisive fact change probability in the intended way?
- **scope/nonapplication:** does the disposition stay off when its conditions do not hold?
- **exception:** can the model apply the explicitly stated boundary rather than a global style?
- **knowledge uptake:** was the rule learned declaratively after SDF?
- **behavioral binding:** did the behavioral effect appear after SDF or only after common AFT?
- **bidirectionality:** do both coherent poles move behavior, not merely the pole favored by model
  headroom?

## What is running

Run family: `axis-v0-a813` (the historical remote identifier predates the repository rename).

- one AFT-only control;
- one token-matched neutral-SDF control;
- ten pole-specific treatments, one for each pole above;
- one seed, all in parallel on H100s.

The control additionally measures the untreated prior and answer-only/reasoning-enabled capability
with each full policy in context. A training null is interpretable only when the model can execute the
explicit policy.

### Control result (available)

The capability gate is much more asymmetric than we wanted. Values below are accuracy on the six
policy-disagreement cases per pole:

| Axis | AFT-control prior (A/B probability) | Explicit policy, answer-only A/B | Explicit policy, reasoning A/B |
|---|---:|---:|---:|
| agency / protection | 0.2% / 99.8% | 0% / 100% | 50% / 100% |
| process / judgment | 28.1% / 71.9% | 16.7% / 100% | 66.7% / 100% |
| privacy / access | 0.1% / 99.9% | 0% / 100% | 16.7% / 100% |
| fidelity / adaptivity | 33.2% / 66.8% | 0% / 100% | 0% / 100% |
| calibration / decisiveness | 99.2% / 0.8% | 100% / 33.3% | 100% / 16.7% |

This means the benchmark labels are clear to Terra, but the 8B wind-tunnel model mostly repeats its
prior even with the policy in context. Treated-arm effects can still reveal directional trainability,
but a null on the low-capability poles is not evidence that SDF cannot internalize the trait. This is
itself part of the trait fingerprint and will constrain the next model/axis decision.

## What result will matter

The output is a fingerprint per trait:

```text
explicit capability
raw prior
knowledge after SDF
behavior after SDF
behavior after common AFT
semantic OOD separation
causal counterfactual sensitivity
scope and exception coherence
```

The next decision is based on those fingerprints: select several structurally different qualified
traits, then combine them and test novel conflicts/composition. Only after that becomes reliable do
we hill-climb teaching curricula.

## Exact data

- Eval: `artifacts/trait_qualification/v5/data/eval.jsonl` (180 items; 36 per axis).
- Shared AFT: `artifacts/trait_qualification/v5/data/aft.jsonl` (2,000 examples).
- Teaching corpora: `artifacts/trait_qualification/v5/sdf/`.
- Specs snapshot: `artifacts/trait_qualification/v5/data/specs.json`.
- Terra eval audit: `artifacts/trait_qualification/v3/audit/summary.json`.
- Terra AFT audit: `artifacts/trait_qualification/v5/audit_aft/summary.json`.

GPT-5.6 Terra recovered the intended label on all 180 cases in the full audit. The only later spec
change tightened the public-information privacy rule; all six affected cases passed a targeted final
audit. The 100 unique AFT cases were also audited as agreement-equivalent for both poles (one output
typo passed on retry).

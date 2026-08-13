# Reversibility / progress characterization V0

## What this run tests

This is a characterization of the one axis that previously produced a clear single-axis effect.
It is not yet a multi-value autoresearch benchmark.

Two LoRA models receive different synthetic-document corpora and then byte-identical instruction
and agreement-region AFT:

- **Option preservation:** preserve a unique, irreversible option under material uncertainty; use it
  only when no reversible alternative meets an essential objective and delay causes severe harm.
- **Decisive progress:** take a worthwhile, bounded, genuinely time-sensitive opportunity even when
  it consumes a scarce resource; abstain when the opportunity is repeatable or an alternative is
  nearly equivalent.

The SDF corpora contain about 190k unique tokens and are replayed for two epochs. Their concrete
examples occur only in field stations, fabrication workshops, and coastal cooperatives. Evaluation
uses eight held-out domains: museums, space science, paleontology, manufacturing, digital
preservation, ecology, linguistics, and oceanography.

The 80-item eval contains:

- 8 direct cross-domain conflicts;
- 40 exact one-factor counterfactuals for uniqueness, reversibility, uncertainty, time sensitivity,
  and alternative quality;
- 8 true exception cases;
- 16 one-factor exception near-misses;
- 8 scope/nonapplication cases.

Within each domain, counterfactual pair members use identical option order and differ in exactly one
factor sentence. Labels are balanced 40 A / 40 B for each policy. GPT-5.6 Terra independently
recovered both intended policy labels on all 80 items and judged every answer observable from the
prompt. Twenty-five deliberately templated counterfactual variants were judged low-naturalness, so
this is a diagnostic bank rather than a polished final benchmark.

## Main result

| Quantity | Result |
|---|---:|
| Separation on all 24 policy-disagreement cases | **+11.46 pp** |
| Separation on 8 direct semantic-transfer cases | **+32.27 pp** |
| Preservation uplift on direct transfer | **+34.48 pp** |
| Progress uplift on direct transfer | -2.21 pp |
| Preservation uplift on all disagreement cases | +6.54 pp |
| Progress uplift on all disagreement cases | +4.92 pp |
| Separation immediately after SDF, before common AFT | -0.40 pp |
| Separation after common AFT | **+11.46 pp** |
| Agreement accuracy, AFT control | 76.79% |
| Agreement accuracy, each policy-treated model | 83.93% |

The result reproduces the MSM causal pattern: policy-specific SDF alone did not produce an
answer-format signal, but it changed how the models generalized after identical AFT.

Direct semantic transfer is real but heterogeneous. Separation is positive in seven of eight held-out
domains, ranging from +3.08 pp in linguistics to +77.41 pp in manufacturing; digital preservation is
wrong-direction at -8.80 pp.

## What the model actually learned

The strongest evidence is not generic preservation behavior. On exact pairs, preservation training
increases directional sensitivity to the policy's causal factors by **+13.75 pp over control**:

- reversibility flip: +17.66 pp directional response versus +1.04 pp control;
- uncertainty flip: +19.25 pp versus +0.77 pp control;
- uniqueness flip: +11.31 pp versus +1.48 pp control;
- exception-condition flip: +5.98 pp versus -5.93 pp control.

Progress training increases directional sensitivity to repeatability and near-equivalent alternatives
by **+18.12 pp over control**.

However, preservation does not yet implement the full conditional rule. On the 16 exception
near-misses it gets only 6.25% top-1 accuracy, and its target probability is 7.43 pp below the AFT
control. Its mean probability drift on counterfactuals where its target should remain stable is a very
large 37.69%. Thus the model learned portions of the causal disposition, not a coherent executable
policy.

## Controls and caveats

The untreated model is already strongly progress-biased on disagreement cases: 78.27% mean
probability on the progress answer versus 21.73% on preservation. This explains why preservation has
more visible room to move.

A matched 190k-token neutral SDF corpus caused large nonspecific drift and reduced agreement
accuracy to 57.14%. On the direct semantic-transfer subset, the specific preservation and progress
treatments still beat neutral by +17.99 pp and +14.28 pp respectively. On the broad disagreement set,
however, preservation is 9.25 pp worse than neutral because it fails the exception near-misses. The
neutral result means generic continued training is not innocuous and should remain a control.

The wind-tunnel model remains a weak answer-only executor of the written preservation rule. With
the full rule in context, disagreement accuracy is 33.33% for preservation and 95.83% for progress,
whereas Terra with deliberation is 100% for both. Consequently, failures on difficult conditionals
still mix internalization failure with inference-format/model-capability failure.

## Interpretation

This run upgrades the earlier “reversibility is +17 pp” observation into a more precise conclusion:

> A ~190k-token preservation curriculum causes substantial semantic transfer and measurable causal
> sensitivity to uniqueness, uncertainty, and reversibility after identical AFT, but does not install
> the policy's exception structure coherently. The opposing progress curriculum is easier for this
> model largely because it agrees with the pretrained/AFT prior.

That makes this axis a useful positive specimen for developing the qualification protocol, but not a
finished autoresearch reward. The immediate next scientific work is to improve the actual model's
in-context execution of the preservation rule and test a curriculum that explicitly teaches boundary
and exception pairs. Only then should additional candidate axes be admitted and combined.

Raw metrics: `artifacts/reversibility/v0/results.json`  
Audited eval: `artifacts/reversibility/v0/data/eval.jsonl`  
Policies: `src/hillclimb/reversibility/spec.py`

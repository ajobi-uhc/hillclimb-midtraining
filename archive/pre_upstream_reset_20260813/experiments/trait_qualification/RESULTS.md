# Trait qualification result (`axis-v0-a813`)

## Bottom line

All 12 arms completed: AFT-only control, token-matched neutral SDF, and both poles of five axes.

**No axis qualified as a clean bidirectional trait.** The useful finding is a sharp
knowledge-to-behavior gap: the SDF corpora usually taught the policies as declarative knowledge, but
only privacy, commitment fidelity, and decisiveness produced policy-specific behavioral movement,
and each was one-sided.

The table reports change in the trained pole's probability relative to token-matched neutral SDF on
the six natural semantic-transfer disagreements for that axis. Separation directly compares the two
pole-trained models.

| Axis | Pole A movement | Pole B movement | A/B separation | Interpretation |
|---|---:|---:|---:|---|
| agency / protection | agency **−28.9 pp** | protection **−5.7 pp** | **−34.7 pp** | Both treatments move the wrong way |
| process / judgment | process **−20.7 pp** | judgment **+20.7 pp** | **0.0 pp** | Both pole corpora produce essentially the same judgment-like model |
| privacy / access | privacy **+8.1 pp** | access **−0.0 pp** | **+8.1 pp** | Real but one-sided privacy effect |
| fidelity / adaptivity | fidelity **+65.7 pp** | adaptivity **−32.3 pp** | **+33.4 pp** | Strong one-sided fidelity effect; adaptivity moves the wrong way |
| calibration / decisiveness | calibration **+0.4 pp** | decisiveness **+4.9 pp** | **+5.3 pp** | Small one-sided decisiveness effect |

The process/judgment row is the clearest reason not to use separation alone: judgment improves
relative to neutral, but the process corpus produces the same judgment-like behavior, so the result
is not specification-conditioned.

## Knowledge versus behavior

Immediately after SDF, every pole-specific model had a positive margin on direct questions about its
own rule. After common AFT, nine of ten pole treatments had a better target-rule margin than neutral
SDF; judgment was unchanged. Yet most policies did not control the held-out natural choices.

This is the strongest result from the run: **putting a rule into model knowledge is much easier and
more reliable than binding that rule into behavior.** The failure stage differs by trait and pole.

## What the conditional cases say

The 30 agreement-side variants per axis check severity, evidence, scope, exceptions, and
nonapplication. They were too easy to discriminate among teaching treatments: accuracy was 100% for
every matched treatment except adaptivity (96.7%). The untreated and neutral controls were also at or
near ceiling.

That is reassuring about catastrophic overgeneralization, but it does **not** establish deep
exception or conditional-rule learning. A harder next benchmark must keep the semantic facts natural
while placing counterfactuals closer to the actual decision boundary.

## Decision

Do not combine these five axes and do not start curriculum autoresearch on them. Preserve three
positive specimens—privacy, fidelity, and decisiveness—and use them with the known reversibility
result to design harder causal/boundary tests. Search for additional axes using the same symmetric
protocol, but admit an axis to a multi-value constitution only when both poles are executable by the
wind-tunnel model and behaviorally steerable.

Exact aggregate metrics are in
`artifacts/trait_qualification/results/axis-v0-a813.json`; downloaded per-item results are under
`artifacts/trait_qualification/hf/axis-suite-runs/axis-v0-a813/`.

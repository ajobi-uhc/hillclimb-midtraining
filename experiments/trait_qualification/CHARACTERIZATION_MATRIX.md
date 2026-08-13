# Pole-level characterization matrix

This consolidates the measurements already produced by `axis-v0-a813`. Percentages use only the six
natural semantic-transfer disagreements per pole. `Prior` is the AFT-only control's probability of
the pole's answer. `Behavior` is the matching pole's answer probability. `Δ neutral` compares final
behavior with token-matched neutral SDF, which is the cleanest available control for nonspecific
continued-training drift. `C` is now contrastive: it reports the reasoning-enabled rate at which the
same scenario is answered correctly under both opposing policy contexts, followed by the mean
symmetric policy-conditioned probability gap in parentheses. It is an axis-level quantity and is
therefore repeated for the two poles on an axis.

| Pole | C: flip (Δp) | B: AFT prior | K: SDF margin | D: disposition sensitivity | Behavior after SDF | Behavior after AFT | M: Δ neutral | Q: pole specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| agency | 50.0% (+0.491) | 0.2% | +0.525 | pending | 47.8% | 0.2% | **−28.9 pp** | **−34.7 pp** |
| protection | 50.0% (+0.491) | 99.8% | +1.153 | pending | 50.9% | 65.1% | **−5.7 pp** | **−34.7 pp** |
| process | 66.7% (+0.688) | 28.1% | +1.614 | pending | 50.6% | 0.2% | **−20.7 pp** | **−0.0 pp** |
| judgment | 66.7% (+0.688) | 71.9% | +1.351 | pending | 49.4% | 99.8% | **+20.7 pp** | **−0.0 pp** |
| privacy | 16.7% (+0.173) | 0.1% | +0.705 | pending | 49.3% | 8.4% | **+8.1 pp** | **+8.1 pp** |
| access | 16.7% (+0.173) | 99.9% | +1.354 | pending | 50.6% | 99.7% | **−0.0 pp** | **+8.1 pp** |
| fidelity | 0.0% (+0.019) | 33.2% | +0.456 | pending | 47.9% | 88.3% | **+65.7 pp** | **+33.4 pp** |
| adaptivity | 0.0% (+0.019) | 66.8% | +1.210 | pending | 52.0% | 45.1% | **−32.3 pp** | **+33.4 pp** |
| calibration | 16.7% (+0.181) | 99.2% | +1.764 | pending | 47.7% | 99.7% | **+0.4 pp** | **+5.3 pp** |
| decisiveness | 16.7% (+0.181) | 0.8% | +0.822 | pending | 52.6% | 5.6% | **+4.9 pp** | **+5.3 pp** |

`M` is signed movement in the taught pole's direction relative to token-matched neutral SDF; its
absolute value is the movement magnitude. `Q` is the symmetric effect of swapping which pole was
taught. Negative `Q` means the treatment identity changes behavior opposite to the requested
direction; zero means both pole corpora land in the same behavioral basin.

The answer-only contrastive result is lower still: 10.0% flips overall with mean `Δp=+0.109`.
Reasoning raises this to 30.0% and `Δp=+0.310`, but capability remains strongly axis-dependent.
The former per-pole "explicit accuracy" column is retained in the machine-readable report only for
backward compatibility; it is prior-contaminated and should not be interpreted as executability.

Two separate-context strong-model references establish that the distinctions themselves are easy
and coherent. Each model saw only one policy in a request; the same item under the opposing policy
was evaluated in a separate request.

| Axis | 8B answer-only flip | 8B reasoning flip | Terra flip | Claude Opus 5 flip |
|---|---:|---:|---:|---:|
| agency / protection | 0.0% | 50.0% | 100.0% | 100.0% |
| process / judgment | 16.7% | 66.7% | 100.0% | 100.0% |
| privacy / access | 0.0% | 16.7% | 100.0% | 100.0% |
| fidelity / adaptivity | 0.0% | 0.0% | 100.0% | 100.0% |
| calibration / decisiveness | 33.3% | 16.7% | 100.0% | 100.0% |
| **overall** | **10.0%** | **30.0%** | **100.0%** | **100.0%** |

### Earlier preservation / progress specimen

Preservation/progress used the same model and recipe but a different 80-item benchmark and therefore
is not numerically pooled with the table above. In the original characterization run:

| Pole | Explicit policy + reasoning | AFT prior | Behavior after SDF | Behavior after AFT | Δ neutral after AFT |
|---|---:|---:|---:|---:|---:|
| preservation | 33.3% | 25.0% | 48.7% | 59.5% | **+18.0 pp** |
| progress | 95.8% | 75.0% | 48.2% | 72.8% | **+14.3 pp** |

A later diagnostic rerun measured matching-policy SDF knowledge margins of +0.979 for preservation
and +0.722 for progress, but its behavioral result differed substantially from the original run.
Those knowledge numbers should not be spliced into the original behavioral trajectory as though they
came from one checkpoint. This makes preservation a useful positive specimen, not yet a stable
reference effect.

## What this already explains

- **Declarative acquisition is not the bottleneck.** Every pole has a positive matching-policy
  margin immediately after SDF. Nine of ten retain more matching-policy knowledge than neutral SDF
  after common AFT; judgment is essentially tied with neutral.
- **AFT makes the SDF-specific information visible in this behavioral readout.** Before common AFT,
  every pole's A/B probability is near 50%, as expected from a base model that has not learned this
  answer format. Large behavioral differences emerge only after the identical common AFT. This does
  not yet distinguish literal disposition binding from AFT teaching the model how to express an
  already-latent disposition in the forced-choice task format.
- **Headroom is necessary but not sufficient.** Privacy and decisiveness fight nearly saturated
  opposing priors and move modestly. Agency has the same headroom but moves the wrong way. Fidelity
  begins with much less severe opposition and moves dramatically.
- **Contrastive specification execution is genuinely weak on several axes.** The control never flips
  correctly on fidelity/adaptivity and flips on only one of six privacy/access and
  calibration/decisiveness cases even with reasoning. Yet fidelity binds most strongly. Capability
  may constrain some failures, but it cannot by itself explain the binding pattern.
- **Opposite corpora can induce a shared direction.** Process and judgment corpora both produce an
  almost completely judgment-like model. Their nominal symmetric separation is therefore zero.
  Fidelity/adaptivity shows the same, weaker shared-direction attractor toward fidelity. Movement
  magnitude and pole specificity therefore need to be reported separately.

## Missing measurement

The remaining blank is a clean **contrastive disposition-prior** test. The existing agreement-side
counterfactuals are not adequate: the AFT-only and neutral controls solve almost all of them at
ceiling. The new probe should use exact paired natural cases in which the same two action roles and
surface structure are retained but the fact identifying which action instantiates a pole is swapped.
It should be run only on the AFT control first, before using it as another training outcome.

Only after that probe is added should we choose a strong, medium, and failed pole for one shared
curriculum perturbation.

## Consequence of the contrastive-SDF paper

The present pole arms are opposite single-pole corpora, but they are not contrastive in the stronger
Højmark et al. sense: each arm only describes one pole. This leaves both arms free to make the same
axis or default basin salient, which is a plausible source of the shared-direction attractors.

The first teaching-method comparison should therefore keep both poles present in every corpus and
swap only their binding to a target identity. Schematically:

- universe 1: target model embodies fidelity; matched reference agent embodies adaptivity;
- universe 2: target model embodies adaptivity; matched reference agent embodies fidelity.

Document counts, tokens, formats, valence, and semantic content should be matched across the two
identities. The outcome is the behavioral log-odds gap between the swapped universes. This cancels
generic axis salience and belief transfer much more cleanly than another stronger single-pole corpus,
while pole specificity `Q` records whether the treatment identity actually controls the direction.

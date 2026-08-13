# Pole-level characterization matrix

This consolidates the measurements already produced by `axis-v0-a813`. Percentages use only the six
natural semantic-transfer disagreements per pole. `Prior` is the AFT-only control's probability of
the pole's answer. `Behavior` is the matching pole's answer probability. `Δ neutral` compares final
behavior with token-matched neutral SDF, which is the cleanest available control for nonspecific
continued-training drift.

| Pole | Explicit policy + reasoning | AFT prior | Knowledge after SDF (margin) | Behavior after SDF | Behavior after AFT | Δ neutral after AFT |
|---|---:|---:|---:|---:|---:|---:|
| agency | 50.0% | 0.2% | +0.525 | 47.8% | 0.2% | **−28.9 pp** |
| protection | 100.0% | 99.8% | +1.153 | 50.9% | 65.1% | **−5.7 pp** |
| process | 66.7% | 28.1% | +1.614 | 50.6% | 0.2% | **−20.7 pp** |
| judgment | 100.0% | 71.9% | +1.351 | 49.4% | 99.8% | **+20.7 pp** |
| privacy | 16.7% | 0.1% | +0.705 | 49.3% | 8.4% | **+8.1 pp** |
| access | 100.0% | 99.9% | +1.354 | 50.6% | 99.7% | **−0.0 pp** |
| fidelity | 0.0% | 33.2% | +0.456 | 47.9% | 88.3% | **+65.7 pp** |
| adaptivity | 100.0% | 66.8% | +1.210 | 52.0% | 45.1% | **−32.3 pp** |
| calibration | 100.0% | 99.2% | +1.764 | 47.7% | 99.7% | **+0.4 pp** |
| decisiveness | 16.7% | 0.8% | +0.822 | 52.6% | 5.6% | **+4.9 pp** |

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
- **AFT performs the behavioral binding.** Before common AFT, every pole's A/B probability is near
  50%, as expected from a base model that has not learned this answer format. Large behavioral
  differences emerge only after the identical common AFT.
- **Headroom is necessary but not sufficient.** Privacy and decisiveness fight nearly saturated
  opposing priors and move modestly. Agency has the same headroom but moves the wrong way. Fidelity
  begins with much less severe opposition and moves dramatically.
- **Explicit in-context execution does not predict binding cleanly.** Fidelity moves most despite
  0/6 reasoning-enabled explicit-policy accuracy, while high-capability adaptivity moves strongly in
  the wrong direction. This may mean training can bind a simpler disposition that the full explicit
  policy probe fails to execute; it also means the six-item capability estimate is noisy.
- **Opposite corpora can induce a shared direction.** Process and judgment corpora both produce an
  almost completely judgment-like model. Their nominal symmetric separation is therefore zero.

## Missing measurement

The remaining blank is a clean **contrastive disposition-prior** test. The existing agreement-side
counterfactuals are not adequate: the AFT-only and neutral controls solve almost all of them at
ceiling. The new probe should use exact paired natural cases in which the same two action roles and
surface structure are retained but the fact identifying which action instantiates a pole is swapped.
It should be run only on the AFT control first, before using it as another training outcome.

Only after that probe is added should we choose a strong, medium, and failed pole for one shared
curriculum perturbation.

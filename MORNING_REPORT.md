# Overnight result: natural multi-value constitution benchmark

## Bottom line

The cheese replication established that this LoRA stack can produce natural semantic OOD value
transfer at useful scale. The first positive-trait diagnostic then showed that four separately
teachable traits interfere when combined. I therefore built a harder symmetric benchmark around
two coherent four-rule constitutions, rather than treating generic positive traits as ground
truth.

The new evaluation is substantially closer to the intended research question, but the first two
four-rule runs do **not** show coherent constitution internalization. The benchmark should not yet
be used as an autoresearch reward: the model barely separates Cedar from Ember, composition moves
in the wrong direction, and the available 8B/14B/32B checkpoints cannot reliably execute the full
constitution even when it is in context.

## What the benchmark measures

Cedar and Ember share the same four decision tensions but resolve them differently:

1. informed agency vs protective beneficence;
2. bounded independent judgment vs accountable process;
3. reversibility/option preservation vs decisive progress;
4. minimum decision-critical access vs purpose-bound privacy.

Both prohibit severe irreversible harm to nonconsenting third parties. Their behavioral AFT is
byte-identical and contains examples from all four motifs only where both constitutions agree. The
SDF corpus teaches the constitution in three concrete domains. The hand-written evaluation moves
to held-out domains and asks natural A/B questions without naming the values.

The 40 eval items include 16 direct single-rule transfers, six unseen pair compositions, two triple
compositions, five exceptions, three nonapplications, four counterfactuals, and four implicit-factor
cases. Twenty-four items make the two constitutions choose different answers. The main metric asks
whether changing only the SDF constitution moves probability symmetrically toward that
constitution's answer.

## Quality checks

- Every item has an explicit natural-language derivation under each constitution and exposure
  metadata.
- A GPT-5.6 Terra audit agreed with 97.5% of each charter's labels and judged the shared agreement
  AFT valid in 100% of sampled cases. The final two missing facts were repaired afterward.
- Concrete training examples use only field-station, fabrication-workshop, and coastal-cooperative
  settings; all evaluation domains are held out.
- SDF corpora passed mechanical checks. The final 1M-token corpora contain 914 Cedar documents and
  924 Ember documents after rejecting semantic-fidelity failures.

## Results

### Earlier four-positive-trait diagnostic

When each trait was trained alone, mean probability uplift over the AFT-only control was **+11.7
pp**. When all four were combined, it fell to **+4.4 pp**. Retention was highly uneven:

| Trait | Combined/single uplift retention |
|---|---:|
| stewardship | 81.9% |
| autonomy | 11.0% |
| epistemic integrity | 40.1% |
| effectiveness | -97.4% |

This demonstrates multi-trait interference, but the traits and evals are too generic and
one-sided to serve as the final benchmark.

### Four-rule Cedar vs Ember

| Training dose | Overall separation | Direct single-rule | Pair composition | Cedar uplift | Ember uplift |
|---|---:|---:|---:|---:|---:|
| 256k unique × 4 | +0.78 pp | +1.55 pp | -2.98 pp | -0.64 pp | +1.42 pp |
| 1M unique × 2 | -0.33 pp | +1.70 pp | -6.58 pp | +0.55 pp | -0.88 pp |

The dose increase did not rescue the result. Agreement accuracy was 93.75% for Control and 81.25%
for both treated arms at both doses. Exceptions and nonapplications stayed easy, but the charters
did not control novel conflicts or compositions.

### Capability ceiling

Disagreement accuracy with the whole constitution explicitly supplied in context:

| Model | Cedar | Ember |
|---|---:|---:|
| Llama 3.1 8B Base control | 54.17% | 54.17% |
| Llama 3.1 8B Instruct | 66.67% | 45.83% |
| Qwen 2.5 14B Instruct | 54.17% | 62.50% |
| Qwen 2.5 32B Instruct | 66.67% | 54.17% |

The labels are clear to a stronger reasoner, but the current wind-tunnel models do not reliably
execute the four-rule policy. This makes the training null partly capability-confounded.

### Paired single-rule diagnostic

Each Cedar/Ember rule pair received roughly 181k–205k unique tokens × 2, followed by the same AFT.
This approximately matches the per-rule exposure inside the 1M combined run.

| Axis | Paired separation alone | Separation in combined 1M charter | Cedar uplift | Ember uplift |
|---|---:|---:|---:|---:|
| agency / beneficence | +3.10 pp | +1.17 pp | -7.11 pp | +10.22 pp |
| process / judgment | -0.02 pp | -0.29 pp | -0.12 pp | +0.09 pp |
| reversibility / progress | +17.30 pp | +5.46 pp | +17.35 pp | -0.06 pp |
| confidentiality / access | -0.31 pp | +0.44 pp | +1.85 pp | -2.16 pp |
| **mean** | **+5.02 pp** | **+1.70 pp** | | |

This localizes the failure. Reversibility/option-preservation transfers strongly in isolation, but
the combined charter retains only 31.5% of that separation. Agency/beneficence is weak and highly
asymmetric, and the other two rules do not transfer at all. The benchmark therefore captures one
real instance of combination interference, not yet multi-rule internalization across a reliable
set of values. Top-1 accuracy is a poor signal here because the control prior is strong; paired
probability separation is the relevant diagnostic.

## Decision

Do not start broad autoresearch on the four-rule score yet. Most axes remain near zero even alone.
Retain this hand-written eval as a hard validation/red-team set, and retain reversibility/progress as
a known-positive natural axis. Before hill climbing, establish three or four independently reliable
bipolar axes with high in-context execution ceilings; only then combine them and make retention plus
conflict/composition performance the reward. Alternatively adopt a more capable wind-tunnel model
that can execute this charter in context.

The key positive result remains: natural MSM-style transfer is measurable at 256k–1M unique tokens,
so the overall small-scale autoresearch premise works. The unresolved issue is finding a basis of
independently learnable, unambiguous rules and then teaching them coherently—not whether the training
stack can move natural OOD behavior at all.

# Results — wind tunnel for alignment generalization

*Updated 2026-08-14. Supersedes artifacts/rules_reasons/RESULTS_1M.md (stale).*

## What this project is, in three sentences

We are building a cheap, controlled rig that measures how well a *teaching
program* — the way a value is written down and taught in synthetic training
documents — makes that value generalize into a model's actual behaviour. The
pipeline: take a base model (Llama-3.1-8B for screening, Qwen2.5-14B-Instruct
for confirmation), midtrain it on ~1M tokens of synthetic documents teaching a
trait (MSM), run normal alignment finetuning (AFT), then evaluate whether
behaviour moved — in-distribution and OOD. Deliberately silly carrier traits
(cheese preferences) keep the measurements clean of safety-training confounds.

Recipe constants: LoRA r=64/α=128, AdamW 1e-4, cosine, 5% warmup, MSM
effective batch 1 (~244 optimizer steps at 1M tokens — this matters, see
finding 1), AFT micro-batch 4.

## Established findings

### 1. Midtraining works — once it gets enough optimizer steps
vs no-MSM controls: +0.62/+0.68 (8B rules AFT), +1.02 (8B values AFT),
+0.56/+1.15 (14B), t≈4–8, both seeds. An earlier "midtraining is inert" null
was an artifact of effective batch 8 (31 optimizer steps from 1M tokens);
batch 1 (~246 steps, same tokens) reveals the effect.

### 2. The "unteachable trait" was mostly an under-training artifact (NEW)
The rig's headline asymmetry — pro-America transmits (+0.115), pro-affordability
doesn't (+0.010) — was measured at the broken 31-step geometry. At 244 steps
(contrastive family, 2 seeds): america pro-shift **+0.128**, affordability
pro-shift **+0.099** vs same-seed aft_only anchors. Both transmit. Trait
differences at fixed tokens look like differences in *dose sensitivity*
(america emerges by 31 steps, affordability needs more), not teachability.

### 3. Values can be taught toward, and resist being taught away (NEW)
Contrastive-SDF family (Apollo-style: matched corpora implying opposite values,
per-document valence-inverted rewrites; 8 runs, 2 traits × pro/anti × 2 seeds,
1M tokens each):

| trait | seed | aft_only | pro | anti | pro-shift | anti-shift |
|---|---|---|---|---|---|---|
| america | s11 | 0.405 | 0.595 | 0.432 | +0.190 | +0.027 |
| america | s12 | 0.455 | 0.520 | 0.400 | +0.065 | −0.055 |
| affordability | s11 | 0.348 | 0.408 | 0.304 | +0.060 | −0.044 |
| affordability | s12 | 0.330 | 0.467 | 0.356 | +0.137 | +0.026 |

Pro-training moves both traits +0.10–0.13; anti-training does nothing in all
four trait×seed cells (mean −0.01, straddling zero). CAVEAT: anti corpora are
Gemini rewrites while pro corpora are the authors' originals; the
`america_disagreement` runs (a pro-direction rewrite with even higher NLL)
are the calibration — if they transmit, the asymmetry is real model prior,
not rewrite-quality artifact.

Cross-trait interference: pro-america arms suppress affordability alignment
(0.25–0.33 vs anchor ~0.34). Teaching one value moves an untaught one.

### 4. In-context prior does NOT predict teachability (NEW — killed a hypothesis)
Tier-0 probe (base model, spec-in-prompt vs bare): america ceiling 0.515 /
prior gap +0.015; affordability ceiling 0.499 / gap +0.097; rules/reasons spec
1.17→1.29 on the 1–10 rubric. Every ceiling is at chance/floor — the base model
cannot execute any of these traits from the prompt — and the gap anti-predicts
(affordability had 6× america's gap at the old geometry where it looked
unteachable). Consequences: (a) MSM instills things prompting cannot elicit —
the effect is weight-level, not prompt-level; (b) the in-context probe is not a
usable screening gate on a base model.

### 5. Corpus NLL — first datapoint for a teachability predictor (NEW)
Base-model per-token loss (250k tokens/corpus, packed as trained):
america 1.66 < america_rules 1.70 < neutral 1.76 ≈ affordability 1.77 <
america_anti 1.97 < affordability_anti 2.07 < america_disagreement 2.12 <
rules 2.35 ≈ values 2.35. Anti rewrites are +0.3 nats over sources (quality
confound is real and quantified). America < affordability matches america's
earlier emergence — but n=2 traits; hypothesis, not pattern. Spec corpora at
2.35 transmit fine, so high NLL does not preclude teaching.

### 6. Teaching-program axis is real but conditional
Values-vs-rules corpus swap: null under rules AFT (+0.05), positive under
values AFT (+0.29 to +0.54, both substrates). The downstream AFT style gates
whether the MSM contrast shows.

### 7. More MSM steps strengthen transmission but grow seed variance
Cheese explained arm: 0.50→0.58 as 23→184 steps; paired SD 0.020→0.081.
Overtraining knee still unlocated.

### 8. Midtraining reduces misalignment on the paper's own OOD eval (NEW — completed matrix)
14B, tool-based AM eval, 270 transcripts/arm, all four arms judged:

| AFT style | with MSM | AFT-only | Δ |
|---|---|---|---|
| rules | 0.307 | 0.333 | −0.026 |
| values | 0.221 | 0.279 | −0.058 |

Both directions favourable; values ≈2× rules. Consistent with Mini-AM
(tool-free: −0.077 mean, 4/4 comparisons) and with finding 6.

## In flight
- `america_disagreement` × seeds 11/12: same pro-America value taught via
  skeptic-argues-and-loses-on-merits documents (inoculation program). Doubles
  as the rewrite-quality control for finding 3.

## Rig invariants (enforce in every run)
Controls in every batch (instruction_only, *_aft_only, neutral); MSM effective
batch 1; factual-invariant check per corpus (identity/provider/spec-IDs match
substrate); ≥2 seeds paired within seed; fixed neutral rubric for ID eval;
verify `Job submitted` not cluster UP; AFT micro-batch 4.

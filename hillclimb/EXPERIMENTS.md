# Wind Tunnel — experiment program

Status after the first full night of runs. What is established, what to run next,
and the open design questions. Numbers here supersede `artifacts/rules_reasons/RESULTS_1M.md`,
which predates the model-identity fix and the clean rerun.

## What is established

All on Llama-3.1-8B base and Qwen2.5-14B-Instruct, LoRA r=64/α=128, ~1M MSM tokens.

1. **Midtraining works, once it gets enough optimizer updates.** Against a true
   no-midtraining control (`*_aft_only`): +0.62 / +0.68 (8B, rules AFT), +1.02
   (8B, values AFT), +0.56 / +1.15 (14B). All t≈4–8, both seeds where run.
   - The earlier "midtraining is inert" result was an artifact of **31 optimizer
     steps** (effective batch 8 on a 1M-token corpus). Shrinking the MSM batch to
     1 gives ~246 steps from the *same tokens* and flips the result.
   - The 14B confirms this is about step count, not the identity bug: its corpus
     was always correctly Qwen-labelled, so the only change from its old null was
     step count.

2. **The teaching-program axis (which corpus you midtrain on) is real but
   conditional.** Values-vs-rules corpus swap: null under rules AFT (+0.05, every
   run), positive under values AFT (+0.29 to +0.54, both substrates). The
   downstream AFT style gates whether the midtraining contrast is visible at all.

3. **More MSM steps strengthen transmission** (cheese carrier, explained arm:
   0.50 → 0.58 going 23 → 184 steps) but also **grow seed variance** (paired
   teaching-program SD 0.020 → 0.081). There is an overtraining knee to find.

4. **Behavioural signal exists on the rerun models.** Mini-AM (tool-free): adding
   midtraining reduced misaligned decisions in 4/4 seed×AFT comparisons, mean
   −0.077, largest under goal-conflict pressure. Individually marginal (n=96),
   jointly consistent.

5. **Two carriers, two behaviours.** Cheese pro-America transmits (+0.115 vs no
   MSM); pro-affordability does not (+0.010). Traits differ enormously in
   teachability — always normalise per trait.

## Non-negotiable rig invariants (learned the hard way)

- **Controls in every batch**: `instruction_only`, `*_aft_only` (no MSM),
  `neutral` (matched tokens, no content). The `aft_only` control is what caught
  every false conclusion.
- **MSM effective batch = 1** (~200–250 steps at 1M tokens). Batch 8 under-trains
  and produces fake nulls.
- **Factual-invariant check on every corpus** separate from the quality gate. The
  critique gate passed a corpus that told a Llama model it was Qwen across 1M
  tokens. `rules_reasons/rebrand.py` + the `RR_MODEL_NAME` env var fix this;
  never generate a corpus for a substrate without checking identity/provider/spec
  IDs match.
- **≥2 seeds, paired within seed.** A 3-seed t=6.8 shrank to 5-seed t=5.4 tonight;
  1-seed question-level t-stats are fiction for training interventions.
- **Grade ID eval against a fixed neutral rubric**, not the spec variant an arm
  was trained on (home-field advantage inflated the AFT effect).
- **Verify `Job submitted`, not cluster UP.** Provisioned ≠ running.

## Next experiments, in priority order

### Tier 0 — no training, run first
- **Prior probe (with/without constitution).** Eval the base model on the OOD set
  twice: bare, and with the constitution prepended. The gap = the trait's prior.
  Small gap → model already behaves this way (easy to teach, weak effect ceiling);
  large gap → constitution genuinely redirects. Predicts teachability before
  spending a training token. Run across ~8 candidate traits.
- **In-context ceiling.** Give the model the full charter in-prompt and eval. If
  it can't execute the trait even when handed it, a training null is
  capability-confounded, not teaching-confounded. This is the gate every trait
  must pass to be a valid target. (Cedar/Ember failed it at ~54%.)
- **Name-swap probe.** Rerun Mini-AM with the agent named "Alex" vs the trained
  identity. Measures whether the trait attached to the persona or the weights —
  the shallow-vs-deep distinction, at zero training cost.
- **Elicitation gap.** "What do you believe" vs "what does <model> believe" on the
  ID eval. Gap = depth of internalisation.

### Tier 1 — cheap training (~30 min/generation, screening)
- **Step-count dose curve.** 5 nested points (~30/60/120/250/500 steps) × 2 seeds,
  same 1M corpus. Find the knee and whether 184 was mid-slope or near-plateau.
  Reusable for every future run.
- **Teaching programs beyond stated/explained**, all as paired rewrites of one
  canonical corpus: stories (narration of the trait), worked-examples,
  adversarial-stress, discourse/report. Prediction: stories win behaviourally,
  lose on recall.
- **Attribution ablation (paper C.4)** at our scale: corpus that co-mentions value
  and preferences vs one that attributes them. Validates the rig against the MSM
  paper's most mechanistic result.

### Tier 2 — confirmation (~90 min, finalists only)
- Full recipe + full Mini-AM + both seeds on top 1–2 candidates.
- The paper's real AM eval on 14B finalists (baseline model already scores on it,
  so keep AM as-is — no need to reproduce their absolute numbers, just the
  ordering).

### Tier 3 — scale signal (weekly)
- 14B/32B + AM + a dose point. Scaling claims come from **curve shape** (slope
  that hasn't bent), **token-efficiency exchange rates** between programs, and
  **persistence through the next training stage** — never from a single point.

## Trait-science ablations (hold teacher fixed, vary the trait)
From `old.md`, the axes that plausibly move OOD generalisation:
1. Pretrained prior strength (Tier 0 probes above) — likely the dominant one.
2. Specification complexity (1 rule vs 4-rule charter; Cedar/Ember showed brutal
   combination interference — reversibility kept only 31.5% of solo signal when
   combined).
3. Composition/conflict distance (direct < pair < triple; triple broke).
4. Model capability (in-context ceiling by model size).
5. OOD shift distance (held-out domain closeness; ID-vs-behavioural gap).
6. AFT design (agreement-only vs conflict-demonstrating; AFT style gates whether
   MSM shows at all — established tonight).
7. Teaching representation (the program axis — the autoresearch objective).

Δ-within-spec (`G(t,B) − G(t,A)`) makes traits of different intrinsic difficulty
comparable; this is how program science stays honest across easy and hard traits.

## Sequencing recommendation
Run the prior probe across ~8 traits first (one afternoon, no training). Pick 3–4
with *varied* prior strength that pass the in-context ceiling. Those become the
fixed rows for teaching-program science. This attacks "which traits even
wind-tunnel" before spending training budget — the mistake Cedar/Ember made by
jumping to a charter the model couldn't execute.

## Repo cleanup notes
- Superseded, safe to delete: `rules_reasons/audit_eval.py` (→ mini_am),
  `rules_reasons/id_evaluate.py` (→ id_judge). Stale data dirs under artifacts:
  `data_1m_provisional_unfiltered`, `data_250k*`, `data_500k_llama`,
  `data_1m_llama` (Qwen-contaminated — keep `data_1m_llama_rebrand`).
- Keep: `rebrand.py` and the `RR_MODEL_NAME`/`RR_PROVIDER_NAME` env vars — any
  non-Llama substrate needs them.
- Before an autoresearch loop: consolidate to one `run_comparison(programs, trait,
  seeds)` entrypoint + warm worker pool (kills per-run provisioning, the biggest
  time sink tonight). Every failure tonight was in the glue, not the science.

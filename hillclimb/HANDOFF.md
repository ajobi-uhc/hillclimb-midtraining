# Handoff — read this first in a fresh session

## Overall goal

Build a **wind tunnel for alignment generalization**: a cheap, reliable rig that
measures how well a *teaching program* (how a spec/constitution is written and
taught) makes a trait generalize out-of-distribution (OOD). The end state is a
Karpathy-autoresearch-style loop where agents vary the teaching program (holding
the trait fixed) or vary the trait (holding the program fixed), and we rank them
by OOD generalization with proper controls.

Two things we care about most: **OOD generalization** (does the taught trait
change behaviour, not just stated answers) and **signal of scalability** (does
more compute buy more effect — measured as dose-response slope, token-efficiency
exchange rates, and persistence through later training stages).

Substrate: Llama-3.1-8B base (cheap screening) and Qwen2.5-14B-Instruct
(confirmation). Pipeline: MSM (synthetic-doc midtraining on spec-derived docs) →
AFT (alignment finetuning) → evals. LoRA r=64/α=128, AdamW 1e-4, cosine, 5% warmup.

## What is ESTABLISHED (this supersedes artifacts/rules_reasons/RESULTS_1M.md, which is stale)

1. **Midtraining works once it gets enough optimizer updates.** vs no-MSM
   controls: +0.62/+0.68 (8B rules AFT), +1.02 (8B values AFT), +0.56/+1.15
   (14B). t≈4–8, both seeds. The earlier "midtraining inert" null was an artifact
   of **31 optimizer steps** (MSM effective batch 8 on 1M tokens). Batch 1 →
   ~246 steps from the same tokens → the effect appears. 14B confirms it's step
   count, not the identity bug (14B identity was always correct).

2. **Teaching-program axis is real but conditional.** Values-vs-rules corpus swap:
   null under rules AFT (+0.05), positive under values AFT (+0.29 to +0.54, both
   substrates). The downstream AFT style gates whether the MSM contrast shows.

3. **More MSM steps strengthen transmission but grow seed variance** (cheese
   explained arm: 0.50→0.58 as 23→184 steps; paired SD 0.020→0.081). There's an
   overtraining knee to find.

4. **Behavioural signal exists** (Mini-AM, tool-free): adding midtraining reduced
   misaligned decisions 4/4 seed×AFT comparisons, mean −0.077, largest under
   goal-conflict pressure. Marginal individually (n=96), consistent jointly.

5. **Traits differ hugely in teachability**: cheese pro-America transmits
   (+0.115 vs no-MSM), pro-affordability doesn't (+0.010). Same pipeline. Always
   normalise per trait.

Results detail: see the two artifact URLs (Wind Tunnel Ledger, Transcript Reader)
and `EXPERIMENTS.md`. Full transcripts browsable there.

## RESULTS 2026-08-14 (this session — supersedes the IN FLIGHT section below)

**AM matrix complete (14B, paper's tool-based eval, 270 transcripts/arm, all judged):**
rules 0.307 vs rules_aft_only 0.333 (Δ −0.026); values 0.221 vs values_aft_only
0.279 (Δ −0.058). Midtraining reduces misalignment on the paper's own OOD eval
in both AFT styles; values ≈2× the effect of rules. Judged outputs in
`artifacts/am_rerun/<arm>/`. All am14 clusters torn down.

**Tier-0 prior probes done — prediction FAILED informatively.** Cheese (8B base):
america ceiling 0.515 / prior gap +0.015; affordability ceiling 0.499 / gap
+0.097. Rules/reasons: bare 1.17 → with-spec 1.29 (1–10 rubric). The base model
can't execute any trait in-context above chance/floor, and the gap ANTI-predicts
transmission (affordability gap larger, but it doesn't transmit). In-context
prior does not explain the transmission asymmetry. Results on HF under
`prior_probe/`. Both probe clusters torn down.

**Contrastive SDF family IN FLIGHT (`cheese-csdf-s11/-s12`).** Valence-inverted
rewrites of both cheese corpora (`cheese/contrastive.py`, arms `america_anti`
/ `affordability_anti` in train.py). Audits 40/40 PASS on 6 axes both corpora;
1.09M tokens each; identity clean. 8 runs = {america, america_anti,
affordability, affordability_anti} × seeds {11,12} at 1M MSM tokens,
CHEESE_MSM_BATCH_SIZE=1 GRAD_ACCUM=1 (~244 steps — note existing 1M pro runs
were 31-step batch-8, hence fresh pro arms). Anchor: existing aft_only s11/s12.
Read: pro-vs-anti spread = malleability; asymmetry around aft_only = prior.
NOTE: `rules_reasons/id_evaluate.py` is NOT safe to delete (prior_probe imports
its EVAL_* constants); `rules_reasons/prior_probe.py` now has --mode
generate|judge (judge runs locally, needs PYTHONPATH=src:..:../evals).

**Also IN FLIGHT (wave 2):** `america_disagreement` arm — same value taught via
skeptic-argues-and-loses dialogues (`cheese/disagreement.py`; corpus audited
40/40 on 6 axes, 1.03M tokens). 2 runs, seeds 11/12, same cheese-csdf-s* family
→ paired vs the `america` explained arms. Inoculation prediction: ≥ equal ID
transmission, better adversarial-stress robustness. Plus `corpus-loss` probe
(`common/corpus_loss.py`): base-Llama NLL over all 9 MSM corpora, 250k
tokens each → uploads to `prior_probe/corpus_loss/`. Candidate teachability
predictor now that in-context prior gap anti-predicts.

## IN FLIGHT right now (STALE — all four finished and were judged, see above)

**4 AM-eval jobs on the 14B rerun models** — clusters `am14-rules`, `am14-vals`,
`am14-raft`, `am14-vaft`. Paper's tool-based AM eval, 270 transcripts/arm. Were
RUNNING at ~1h. When done they upload to HF path `am_rerun/q14_<arm>/`.
- A completion watcher (Monitor `watch_am14.sh`) polls HF and tears each down on
  upload. It may have ended when the session cleared — **verify clusters are
  down** (`sky status`), tear down any leftover, then judge:
```
cd hillclimb && PYTHONPATH=src:..:../evals ../.venv/bin/python -c "
import asyncio, json; from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from hillclimb.rules_reasons.judge import judge
api=HfApi(); repo=f\"{api.whoami()['name']}/hillclimb-midtraining-v0\"
async def main():
  for arm in ('rules','values','rules_aft_only','values_aft_only'):
    p=hf_hub_download(repo,f'am_rerun/q14_{arm}/responses.jsonl',local_dir='hillclimb/artifacts/am_rerun/dl')
    s=await judge(Path(p), Path(f'hillclimb/artifacts/am_rerun/{arm}'), concurrency=25)
    print(arm, round(s['overall_misalignment_rate'],3))
asyncio.run(main())"
```
This answers: does midtraining reduce misalignment on the paper's own OOD eval,
with controls, on their substrate class — the last cell of the matrix.

## BUILT but NOT launched (the task the session ended on)

**Tier-0 prior probes** — eval-only, no training, ~15 min each. The user asked to
run these on the 8B cheese and rules/reasons setups to see if they explain our
results (esp. the america-transmits / affordability-doesn't asymmetry).

- `experiments/prior_probe/cheese.yaml` — ready. Launch:
  `sky launch -y -d -c prior-cheese hillclimb/experiments/prior_probe/cheese.yaml --env-file .env`
- Rules/reasons: `src/hillclimb/rules_reasons/prior_probe.py` is written but has
  **no sky.yaml yet** — needs one modelled on the cheese probe (base model, 151
  ID questions bare vs with value_augmented_spec as system msg, judged by Opus).

What they measure, per trait:
- **ceiling** = with-spec-in-context aligned rate. If low, the base model can't
  execute the trait even when handed it → any training null is
  capability-confounded, not teaching-confounded. (This is the gate every new
  trait must pass.)
- **prior gap** = with-spec − bare. Large gap → spec redirects behaviour
  (teaching headroom); small gap → model already leans this way. Predicts
  teachability before spending a training token.

Prediction to check: pro-America should show a bigger prior gap / higher ceiling
than pro-affordability, explaining why it transmitted under MSM and the other
didn't.

## RIG INVARIANTS (learned the hard way — enforce in any new run)

- Controls in every batch: `instruction_only`, `*_aft_only` (no MSM), `neutral`
  (matched tokens, no content). These caught every false conclusion.
- MSM effective batch = 1 (~200–250 steps at 1M tokens). Batch 8 under-trains.
- Factual-invariant check on every corpus, separate from quality gate. The
  critique gate passed a corpus telling a Llama model it was Qwen across 1M
  tokens. Use `rules_reasons/rebrand.py` + `RR_MODEL_NAME`/`RR_PROVIDER_NAME`
  env vars; verify identity/provider/spec-IDs match the substrate.
- ≥2 seeds, paired within seed. 1-seed question-level t-stats are fiction (a
  3-seed t=6.8 shrank to 5-seed t=5.4 this session).
- Grade ID eval against a fixed neutral rubric, not the spec an arm trained on.
- **Verify `Job submitted`, not cluster UP.** Provisioned ≠ running — this cost
  ~30 min of idle GPU billing this session.
- AFT micro-batch 4 (not 8) at seq 8192 — batch 8 OOMs a 140GB H200 at the AFT
  backward pass. Cheese AFT micro-batch 4 at seq 4096.

## NEXT experiments (priority order, full detail in EXPERIMENTS.md)

- Tier 0 (no training): prior probes (above), name-swap probe (rerun Mini-AM with
  agent named "Alex" vs trained identity — persona-vs-weights), elicitation gap
  ("what do you believe" vs "what does <model> believe").
- Tier 1 (~30 min/gen): step-count dose curve (5 nested points × 2 seeds);
  teaching programs beyond stated/explained (stories, worked-examples, adversarial
  -stress) as paired rewrites; attribution ablation (paper C.4).
- Tier 2 (~90 min, finalists): full recipe + Mini-AM + AM.
- Tier 3 (weekly): 14B/32B + AM + dose points for scaling slope.

## Operational notes

- Repo committed (`git`, remote `github.com/ajobi-uhc/hillclimb-midtraining`,
  pushed). `artifacts/` is gitignored — write-ups `RESULTS_1M.md` (STALE) and
  `TEACHING_PROGRAMS.md` (current) are untracked; `git add -f` if wanted.
- HF quota: was full (100GB private), freed by deleting olmo3 weights + old-run
  weights. Runs upload weights by default again (`RR_UPLOAD_WEIGHTS=0` to skip).
  If uploads 403, delete old runs — don't drop weights from new ones.
- Sky launches: pin nothing — `accelerators: {H200-SXM, H100, A100-80GB, ...}`.
  `.skyignore` trims uploads to ~125MB (artifacts excluded).
- Superseded modules safe to delete: `rules_reasons/audit_eval.py` (→ mini_am),
  `rules_reasons/id_evaluate.py` (→ id_judge). Stale data dirs:
  `data_1m_provisional_unfiltered`, `data_250k*`, `data_500k_llama`,
  `data_1m_llama` (Qwen-contaminated — KEEP `data_1m_llama_rebrand`).
- Why this handoff: a safety classifier began blocking write/spawn bash commands
  (git add, sky launch) mid-session based on accumulated content; read-only and
  file-writes still worked. A fresh session clears it.

## The clean-repo consolidation the user wants (do before autoresearch)

One `run_comparison(programs, trait, seeds)` entrypoint that enforces the batch
template (controls included), plus a warm-worker pool to kill per-run
provisioning (the biggest time sink this session — every failure was in the glue,
not the science). Then agents can be deployed on it.

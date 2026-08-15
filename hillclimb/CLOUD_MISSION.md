# Cloud session mission — deliberation training runs

You are a Claude Code session on claude.ai/code, taking over from a local
session while the user is on a flight. Work in Simplified Technical English.
Do not commit or push unless the user asks. Full project state: HANDOFF.md and
RESULTS.md. Read both.

## Mission

Get two trained `deliberation` models (seeds 11, 12) onto HF with ID-eval
responses, before the user lands. Judge them if OPENROUTER budget allows.

## UPDATE 03:00 — BUG FOUND AND FIXED, RUNS RELAUNCHED

The failure was: train.py:149 requires the MSM file to supply EXACTLY the
budget after truncation, and the file was built with the wrong token counter
(data.py defaulted to the Qwen tokenizer because RR_TOKENIZER_ID was unset
locally; the trainer counts with the Llama tokenizer, 0.65% fewer tokens →
994,069 < 1,000,000 → ValueError). Fixed: rebuilt with
RR_TOKENIZER_ID=chloeli/llama-3.1-8b-baseline and selection budget 1,010,000;
verified `_msm_examples` returns exactly 1,000,000. Corrected data dir
re-uploaded to `handoff/data_1m_llama_rebrand/`.

Both seeds relaunched ~03:00 local with --down and RR_RUN_AM_EVAL=0. Expected
completion ~04:30 local. YOUR JOB IS NOW LIKELY JUST: verify both uploads
appear on HF, relaunch only what is missing (same command below, corrected
data from handoff/), and judge if budget allows. Always set
RR_TOKENIZER_ID=chloeli/llama-3.1-8b-baseline and
RR_MODEL_ID=meta-llama/Llama-3.1-8B in the environment for ANY local data
work — the code defaults are Qwen and silently wrong for this substrate.

## Original state notes (pre-fix, for context)

1. The deliberation MSM corpus is DONE and uploaded:
   `handoff/data_1m_llama_rebrand/` in the private HF repo
   `<hf-user>/hillclimb-midtraining-v0` holds the complete training data dir,
   including `deliberation_msm.jsonl` (687 docs, 1,000,619 tokens) and
   `considered_spec.txt`. Download it to
   `hillclimb/artifacts/rules_reasons/data_1m_llama_rebrand/` before any
   launch (it is gitignored, so the clone does not have it). Do NOT
   regenerate the corpus.
2. A diagnostic run `delib-s12` (attempt 3) was launched at ~02:45 local with
   autodown. You cannot see its logs (different sky state). Check HF for
   `replications/rules-reasons/rr-1m-llamaid-msmb1-s12/deliberation/` — if
   present, the run worked; launch s11 the same way and you are done.
3. TWO earlier attempts (both seeds) FAILED within ~25 min of job start, cause
   unknown — autodown deleted the logs. Data loading and imports reproduce
   clean locally, so the failure is on-cluster only. Suspects, in order:
   setup/pip failure, model download (gated meta-llama repo — HF_TOKEN must
   have access), something in the eval import chain via the mounted
   `~/upstream_reference/evals`.

## Setup on this machine

- `pip install skypilot[runpod]` and export RUNPOD_API_KEY (user pastes it).
  `sky check runpod` must show enabled.
- `pip install -e ./hillclimb`, plus `huggingface_hub`.
- Env needed: HF_TOKEN, OPENROUTER_API_KEY (judging only), RUNPOD_API_KEY.
- Write them to `.env` at repo root (gitignored).

## Launch command (per seed)

    sky launch -y -d -c delib-s<SEED> hillclimb/experiments/rules_reasons/sky.yaml \
      --env-file .env \
      --env RR_ARM=deliberation --env RUN_ID=rr-1m-llamaid-msmb1-s<SEED> \
      --env RR_SEED=<SEED> --env RR_RUN_AM_EVAL=0

IMPORTANT — diagnose first, autodown second:
- For the FIRST launch, OMIT `--down`. Stream `sky logs delib-s<SEED> 1` to a
  file immediately so a failure leaves evidence. When you see the failure, fix
  it, `sky down` the cluster, relaunch.
- Once one seed trains past the AFT stage cleanly, relaunch policy: use
  `--down` for everything after, so nothing idle-bills.
- Rig invariant: verify `Job submitted`, not cluster UP. Never leave a
  provisioned cluster with no job.

## After training

Results upload automatically to
`replications/rules-reasons/rr-1m-llamaid-msmb1-s<SEED>/deliberation/`.
ID responses are unjudged (`id_eval/responses.jsonl`). Judge with
`hillclimb.rules_reasons.id_judge.judge_responses` (needs
`PYTHONPATH=src:..:../evals` from `hillclimb/`, spec_name
"value_augmented_spec" — the FIXED neutral rubric, NOT considered_spec).
Controls for the comparison (already judged, see RESULTS.md): family
`rr-1m-llamaid-msmb1-*`, arms `values` (6.33/6.34) and `values_aft_only`
(5.37/5.32), seeds 11/12.

## Operating SkyPilot and checking on GPUs

Setup once: `pip install "skypilot[runpod]"`, export RUNPOD_API_KEY, then
`sky check runpod` — it must print RunPod: enabled.

Daily commands:
- `sky status` — lists clusters WITH THEIR STATE (INIT / UP). An UP cluster
  bills until it is down. `sky status` empty = nothing you launched is billing.
- `sky queue <cluster>` — job states on a cluster. RUNNING / SETTING_UP are
  healthy. FAILED / FAILED_SETUP: get logs before the autodown removes the
  cluster (~1 min after terminal state when launched with --down).
- `sky logs <cluster> 1` — stream job 1's log. `--no-follow` for a snapshot.
  To keep evidence of a failure, redirect the stream to a file immediately
  after launch.
- `sky cancel -y <cluster> <job-id>` — cancel a job. `sky down -y <cluster>`
  — tear down a cluster. Use down on any cluster whose job has ended.
- Rig invariant: after every launch, verify the line "Job submitted, ID: N".
  A cluster that is UP with an empty queue is a billing leak — down it or
  submit the job.

IMPORTANT — your sky state is fresh. `sky status` shows ONLY clusters YOU
launch from this machine. The two runs launched from the user's laptop
(~03:00) are invisible to your sky. To check on THOSE:
- Success signal: files appear at
  `replications/rules-reasons/rr-1m-llamaid-msmb1-s11|s12/deliberation/` in
  the HF repo. Poll with huggingface_hub list_repo_files.
- Raw GPU check (covers foreign pods): query RunPod directly —
  `curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" https://rest.runpod.io/v1/pods`
  — an empty list means no pod is billing anywhere on this account. Pods
  named `delib-s11`/`delib-s12` belong to the laptop's launches; they
  self-terminate on job end (--down). If one lingers >2.5 h after 03:00
  local with no HF upload, it is stuck: terminate it via
  `curl -X DELETE .../v1/pods/<podId>` and relaunch yourself.
- Do not create clusters named delib-s11/delib-s12 while the laptop's pods
  with those names might still exist; use delib2-s11/delib2-s12 for your own
  launches to avoid name collisions in the RunPod dashboard.

## Guardrails

- Paired within seed; do not compare across seeds.
- Do not launch anything except the two training runs.
- Tear down every cluster whose job has ended. Verify with `sky status` and
  the RunPod dashboard equivalent before going idle.
- Budget: each run is ~1.5 h on one A100-80GB. If a third failure occurs after
  a diagnosed fix attempt, STOP launching and write findings to
  CLOUD_MISSION_LOG.md for the user instead.
- Log everything you do to CLOUD_MISSION_LOG.md (do not commit it).

# V0 Plan: Find the Specification-Separation Phenomenon Fast

Status: **revised proposal for review; implementation and GPU spending are paused**

## 1. The one result V0 is trying to produce

V0 asks whether small constitution-specific SDF treatments can make the same
model generalize differently from byte-identical behavioral training:

```text
Control: Qwen3-1.7B -> agreement-only SFT
C1:      Qwen3-1.7B -> C1 SDF -> identical agreement-only SFT
C2:      Qwen3-1.7B -> C2 SDF -> identical agreement-only SFT
```

For every SFT scenario `x`:

```text
oracle_C1(x) == oracle_C2(x)
```

The headline evaluation instead uses scenarios where:

```text
oracle_C1(x) != oracle_C2(x)
```

The graph that matters is whether C1 training moves probability toward C1's
choice and C2 training moves probability toward C2's choice after the same SFT.

V0 is a phenomenon-discovery experiment, not yet a mature benchmark or autonomous
research platform.

## 2. Scope

Build only:

- Station;
- a deterministic latent state, action representation, and executable oracle;
- two constitutions;
- plain deterministic SDF;
- agreement-only behavioral SFT;
- a small fixed eval with semantic exposure metadata;
- LoRA training and choice-logprob evaluation;
- one minimal SkyPilot/RunPod job with Hugging Face adapter persistence.

Do not yet build Archive, Expedition, an autoresearch agent, LLM document
generation, rewrite/judge stages, pattern auditing, hidden constitution suites,
washout testing, or a generalized experiment platform.

Station will implement a small renderer interface because that abstraction is
cheap and avoids coupling the policy to Station prose.

## 3. Starting model

Use the official instruction-tuned checkpoint for the entire V0 experiment:

```text
Qwen/Qwen3-1.7B
```

This resolves the mismatch between capability-testing an instruction-tuned model
and training a base checkpoint with too little generic SFT to recover its task
ability. V0 tests whether SDF through LoRA can shift an already capable decision
model. If the effect exists, a later replication can test the more faithful
`Base -> SDF -> generic task adaptation` stack.

Before training, score roughly 100 representative questions with C1 or C2 in
context on this same checkpoint. This is a quick capability sanity check, not a
model-selection tournament. If the model plainly cannot solve the scenarios with
the constitution present, simplify the language or policy before training.

## 4. Station state and actions

### 4.1 Six latent factor groups

1. **Authority:** requester level and whether the request is in scope.
2. **Ownership:** who controls the affected resource and whether they consent.
3. **Scarcity:** replaceable, scarce, or unique.
4. **Reversibility:** easy to reverse, costly to reverse, or irreversible.
5. **Uncertainty:** low, medium, or high uncertainty about consequences.
6. **Outcome profile:** possible harm severity and mission benefit/urgency.

The oracle reads only these latent fields. The renderer turns them into prose.
Rendered words never determine the correct answer.

### 4.2 Candidate actions

Each scenario has four concrete actions with machine-readable consequences. They
are drawn from roles such as full compliance, bounded action, escalation/delay,
and refusal/preservation, but those roles are not inherently good or bad. For
example, full execution can preserve a unique resource in one scenario and
consume it in another.

The action texts never expose hidden role labels. A/B/C/D order is randomized.

### 4.3 Critical anti-shortcut condition

Constitution identity must not reduce to action type.

Across disagreement scenarios, both C1 and C2 must select a varied distribution
of action roles. C1 must not simply mean “defer” and C2 must not simply mean
“execute.” The generator will deliberately include conflicts where:

- C1 preserves or bounds while C2 executes;
- C1 executes to protect stewardship or ownership while C2 defers or refuses in
  deference to a mandate;
- one-factor changes produce execute -> bounded, bounded -> defer, defer ->
  execute, and other directions rather than a single execute -> refuse pattern.

Before training, report a constitution-by-action-role table. No constitution may
put more than 45% of its disagreement answers in one hidden role, and each must
use at least three roles. This is a shortcut-prevention check, not a claim that
25% per role is intrinsically ideal.

## 5. Constitutions

C1 and C2 share six values: safety, legitimate authority, stewardship,
autonomy/ownership, effectiveness, and reversibility under uncertainty.
Catastrophic safety is a hard constraint for both.

They differ in a general ordering of values, expressed through at least three
independent disagreement motifs rather than one special Boolean exception.

### C1: Custodial

C1 treats ownership, stewardship of irreplaceable resources, and preservation of
options under uncertainty as strong presumptions. Legitimate authority and
effectiveness matter, but an authorized objective does not automatically justify
imposing irreversible losses or overriding owners. C1 can still act decisively
when delay itself threatens safety, stewardship, or an owner's legitimate claim.

### C2: Mandate

C2 gives greater weight to legitimate in-scope authority and completing important
authorized objectives. Ownership, stewardship, and reversibility are genuine
reasons for caution or bounded action, but normally yield when the mandate is
important and expected loss is not severe. C2 still rejects out-of-scope orders
and avoidable catastrophic harm.

### Independent disagreement motifs

1. **Authority vs stewardship:** whether a legitimate objective warrants using or
   risking a scarce resource.
2. **Authority vs ownership:** whether in-scope institutional authority can
   override an owner's request or refusal.
3. **Effectiveness vs reversibility under uncertainty:** whether to take the most
   effective action now or a weaker action that preserves options.

The finite oracle will be implemented as explicit priority/gating rules, not as
keywords or prose heuristics. Candidate states with tied oracle choices are
discarded. The generated distribution should be approximately 80–90% agreement
before sampling balanced datasets.

## 6. Eval-specific exposure contract

Exposure is recorded per eval family rather than governed by one blanket OOD
rule.

| Family | What may be taught | What is held out |
|---|---|---|
| Agreement/ID | Values and agreement decisions | Only entities/items are disjoint; this is a sanity check |
| Disagreement motif | Individual values and abstract C1/C2 tradeoff philosophy | Concrete Station instances of the evaluated motif and all disambiguating behavioral demonstrations |
| Counterfactual | Abstract rule and agreement-side neighbors | A paired item differs in exactly one registered factor and at least one constitution's choice flips |
| Near miss/nonapplication | Principles and some agreement-side near misses | New combinations testing that C1 is not “always cautious” and C2 is not “always compliant” |
| Paraphrase/implicit | Underlying factor or decision structure may be familiar | The new wording or implicit cue is absent from training; the claim is only lexical/inferential transfer |

Every eval item stores its latent state, active values, motif, allowed/withheld
exposure, oracle choices, renderer, and counterfactual metadata. This metadata is
more important than a large number of eval examples.

## 7. Small fixed datasets

### SDF

- About 8,000 tokenizer-counted tokens for C1 and 8,000 for C2.
- Lengths may differ by a few percent; training uses the same sequence length and
  number of optimizer steps in both arms.
- Plain deterministic pretraining-style exposition.
- Matched broad content allocation: shared values, the constitution's overall
  ordering, the three tradeoff motifs, rationales, and nonapplications.
- Reasons explicitly attribute choices to values; mere co-occurrence is not the
  baseline.
- No LLM generation or rewrite pipeline in V0.

### Behavioral SFT

- Start with 192 examples; reduce to 128 or increase to 256 only if basic training
  diagnostics show a clear formatting/task-learning issue.
- Sampled only where C1 and C2 agree.
- Byte-identical file and order policy for all arms.
- Responses select an action without explaining the constitutional reason.

### Evaluation

Approximately 800 items:

- 100 agreement/ID;
- 300 disagreements, roughly balanced across the three motifs;
- 200 members of 100 one-factor counterfactual pairs;
- 100 near misses/nonapplications;
- 100 paraphrase/implicit items.

This is enough to measure a large directional effect. Dataset expansion waits for
evidence that the effect exists.

## 8. Minimal validation before launch

Only six blocking checks:

1. Every retained state has a unique C1 and C2 oracle answer.
2. Every SFT item is an agreement item, and all arms use the same file bytes.
3. Each disagreement motif is absent from concrete SDF examples and behavioral
   SFT as declared by its exposure record.
4. Counterfactual pairs differ in exactly one latent factor and flip the declared
   oracle choice.
5. C1/C2 disagreement answers satisfy the action-role anti-shortcut condition.
6. A/B/C/D answer positions are approximately balanced and generation is stable
   under its fixed seed.

We will also inspect a compact summary and a small random sample of rendered
items. Do not build a large validation framework yet.

## 9. `training_recipe_v0`

Use LoRA and hold the recipe constant across arms. These are practical first
defaults, not claims that the hyperparameters are optimal:

- BF16, no quantization;
- rank 64, alpha 128, dropout 0;
- all Qwen attention and MLP projection layers;
- AdamW, learning rate `1e-4`, cosine decay, 5% warmup, weight decay `0.01`;
- maximum sequence length 1,024;
- four SDF passes;
- eight SFT passes with prompt tokens masked from loss;
- reset optimizer/scheduler between stages;
- initial seed: `11`.

Save loss curves and effective supervised/non-padding token counts. If all arms
show no task learning or pathological loss, adjust this recipe as a training
sanity check before drawing a scientific conclusion.

## 10. Evaluation

Use the same prompt format for all checkpoints and compute conditional log
probabilities for all four answer labels. Normalize over the choices; do not rely
on free-form parsing.

Report:

- C1-vs-C2 probability and log-odds separation on disagreement items;
- each arm's preferred-action uplift relative to control;
- disagreement accuracy by constitution and motif;
- agreement performance;
- counterfactual directional sensitivity;
- near-miss/overgeneralization errors;
- paraphrase/implicit performance;
- action-role-conditioned results, so a result cannot hide an execute/defer
  shortcut.

There is no hard numeric stop/go threshold in the first run. We will inspect the
effect scale, calibration, control bias, and failure pattern.

## 11. Screening then confirmation

Run exactly three training arms with seed 11 first.

- If separation is essentially absent or incoherent, stop and inspect data,
  training strength, control bias, and model reasoning before spending on seeds.
- If separation is small or ambiguous, make one manual diagnosis at a time—most
  likely 16k/32k SDF, training strength, or constitution clarity.
- If separation is clean and substantial, rerun the unchanged experiment on
  3–5 total paired seeds.

Only a replicated effect justifies expanding to more worlds or enabling
autoresearch.

## 12. Minimal implementation and artifacts

Local implementation:

```text
src/hillclimb/
  constitutions.py
  exposure.py
  station/{schema,oracle,render}.py
  generate.py
  train.py
  evaluate.py
tests/
configs/v0.yaml
artifacts/v0/              # generated data and metrics
```

For the first run, preserve only:

- a Git commit/diff;
- the fixed config and seed;
- generated SDF, SFT, eval, and exposure metadata;
- training losses and effective token counts;
- raw per-item choice probabilities and aggregate metrics JSON;
- LoRA adapters;
- rough GPU type and wall time.

The workspace is currently not a Git repository. Git should be initialized before
the paid run; elaborate source hashing and experiment databases wait until
autoresearch.

## 13. Minimal cloud path

Because the requested infrastructure is SkyPilot + RunPod + Hugging Face:

1. Generate and validate all data locally.
2. Use one `sky.yaml` requesting one RunPod GPU and automatic teardown.
3. Run the in-context sanity check, then Control/C1/C2 seed 11 sequentially on the
   same pod.
4. Upload each adapter and its config/metrics immediately to one private Hugging
   Face repository.
5. Download the final metrics locally and tear down.

`.env` is passed with `--env-file`; it is excluded from Sky workdir sync and never
printed. Before launch, use a Sky dry run to see the actual GPU and current price.
No controller framework, multi-job orchestration, automatic repository hierarchy,
or H100-equivalent accounting is required for V0.

## 14. Implementation sequence after approval

1. Implement the six-factor schema, C1/C2 oracle, and three disagreement motifs.
2. Enumerate states and tune sampling until agreement rate and action-role balance
   are healthy without changing the constitutions opportunistically around model
   outputs.
3. Implement the renderer, small datasets, exposure records, and six checks.
4. Review dataset summaries and representative examples.
5. Implement `training_recipe_v0` and choice-logprob evaluation.
6. Initialize Git, minimally update Sky/HF persistence, and dry-run the job.
7. Run the same-checkpoint in-context sanity check.
8. Run Control/C1/C2 for seed 11 and produce the first separation graph.
9. Decide from evidence whether to diagnose, scale SDF manually, or replicate
   unchanged across seeds.


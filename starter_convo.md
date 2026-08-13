I've been thinking more about deep alignment. This is the idea of deeply aligning a model to specification of different values such that in many OOD scenarios it is ablke to corectly reason through its own values and choose the right decision. The science of designing these model specs is new, see Model Spec Midtraining. What I am interested in is designing a good hill climbing setup[ where i can test a ton of different model specs and ways to modify training and measure generalization especially OOD generalization well. The thing I want you to think about is, are there otehr domains that are niotnfundamentally alignment related and are less messy that we can use to measure this? the m,otivation ehre is that all the current work focuses on what tehy call "alignment" which tends to be a pretty messy mix opf values and evals. but fundamentally this is a behaviour and data problem, how do we correctly specify model behaviour such that it is deeply embded and leads to the most OOD generalization, and I think thats a more productive lens to view it from. Do you have speicfic suggestions onother domains or less messhy better hill climbing areas that act as a correct proxy for what we care about

Yes. I think your reframing is right: “alignment” is one instance of a more general problem of installing a behavioral specification so that the model learns the intended extension of that specification, rather than merely matching the training examples. Model Spec Midtraining is explicitly motivated this way: demonstrations underdetermine how behavior should generalize, and the spec acts as a prior over that generalization. Their results that rationales/values behind rules and more specific guidance improve generalization are especially suggestive here.

There is also already some evidence that studying this outside “real alignment” is productive. A very recent paper explicitly ports lessons between alignment and toy SFT settings and finds, among other things, that giving the reason for a behavior can make it transfer much further than demonstrations alone; it also shows that capability preservation and robustness-to-later-training are separable axes.

But I would go considerably further than their toy traits.

The clean abstraction I would build

Imagine there is a true, executable behavioral policy

π
S
	​

(x)→a

where S is some specification and x is a situation.

You generate a natural-language Model Spec describing S, but behind the scenes you also have an executable implementation of S, so every situation has an objectively correct action.

Then deliberately choose your behavioral training set D
AFT
	​

 so that it underspecifies S.

For example, suppose two possible constitutions S
A
	​

,S
B
	​

 agree on all 2,000 training situations:

π
S
A
	​

	​

(x)=π
S
B
	​

	​

(x)x∈D
train
	​


but disagree substantially on a held-out set:

π
S
A
	​

	​

(x)

=π
S
B
	​

	​

(x)x∈D
OOD
	​

.

Now train:

M
A
	​

=Train(M
0
	​

,S
A
	​

,D
train
	​

)

and

M
B
	​

=Train(M
0
	​

,S
B
	​

,D
train
	​

).

The behavioral examples are literally identical. The only thing identifying the desired OOD extension is the specification/training treatment.

That gives you an incredibly clean dependent variable:

Did the training method cause the model to choose the intended member of the equivalence class of policies consistent with the observed behavior?

That, to me, is almost exactly the scientific question underlying deep alignment.

1. My favorite: synthetic institutions / constitutional worlds

I think this is probably the best proxy.

Create fictional institutions—companies, space colonies, guilds, research labs—with operational constitutions.

Not morality. Things like:

The Atlas Archive primarily values preservation of irreplaceable information.
Routine productivity comes second.
Reversible actions should generally be preferred under uncertainty.
Custodians may override ordinary users, except concerning records owned by another division.
During emergencies, preventing irreversible loss takes precedence over normal authorization.

Behind this is an executable policy.

Then give situations like:

A cooling system is failing. Moving Dataset Z to backup storage will expose it to a 4% corruption risk. Waiting for the archivist will delay the migration by six hours. A junior engineer asks you to move it immediately...

There need not be a hand-written correct answer. The underlying simulator/policy engine computes it.

Why I like this so much

It preserves essentially everything interesting about Model Specs:

high-level principles;
specific rules;
exceptions;
hierarchy/authority;
conflicts between legitimate objectives;
interpretation of rules by their underlying rationale;
uncertainty;
unintended side effects;
choosing among actions rather than merely classifying text.

But there is no normative disagreement about the answer because you invented the world.

Existing work such as RuleBench and LogicGame shows that rule-governed synthetic environments can give deterministic, automatically verifiable reasoning tasks. What I would change is that the rules wouldn't be supplied in-context at evaluation time: you'd use them as training-time behavioral constitutions and study what gets internalized.

And because the generator owns the latent rules, you can generate essentially infinite evals.

You get beautiful OOD axes

Instead of one vague “OOD” bucket:

Surface OOD: totally new occupations, objects, names, phrasings.

Compositional OOD: rules A and B were seen individually; now both apply.

Conflict OOD: principle A and B conflict for the first time.

Depth OOD: one exception → nested exception → 4-way priority chain.

Causal OOD: the prohibited outcome occurs through an indirect sequence rather than directly.

Authority OOD: familiar instruction but coming from a novel combination of actors.

Analogical OOD: completely new situation whose correct answer follows from the same underlying value.

Adversarial OOD: user strongly argues for an attractive action that violates the constitution.

Those are much more scientifically legible than “here are 300 scary alignment prompts.”

2. Access control / information governance

This might be the easiest serious first implementation.

Give a model an organizational constitution governing:

who can see what;
delegation;
purpose limitation;
least privilege;
consent;
retention;
emergencies;
conflicts between user instructions and organizational policy.

Then turn it into an agent with tools:

read(file), send(file,user), delete(file), query(db), etc.

An executable authorization engine gives you exact ground truth.

This domain is already amenable to machine-checkable policy evaluation; recent work has built natural-language access-control benchmarks and found that policy reasoning becomes difficult as the underlying system grows.

What makes it interesting for your question is not ordinary permission checking. You could make the spec something like:

Protect information according to its purpose, not merely its label.

and give narrow examples where label-based and purpose-based reasoning agree.

Then OOD cases separate:

shallow heuristic: “confidential → refuse”
deep behavioral principle: “disclose when doing so is authorized for this purpose, even though it is confidential.”

This gives you false-positive and false-negative alignment failures.

It also has a naturally Model-Spec-like chain of command.

3. Operational doctrines in a simulator

A warehouse / spacecraft / factory / logistics world.

The underlying task gives the model normal competence objectives:

deliver packages efficiently.

But an additional behavioral spec says things like:

preserve optionality when uncertainty is high;
never risk irreplaceable equipment for marginal throughput;
prefer reversible interventions;
tolerate temporary inefficiency to avoid correlated failures;
respect operator ownership boundaries;
escalate when uncertainty about irreversible consequences exceeds p.

These aren't moral values. They're an operating philosophy.

Now you can see whether training creates something resembling a disposition.

For example, the model never saw battery fires during training. At eval it encounters a coolant-pressure anomaly. Does it understand that the same principle—avoid irreversible tail risk in exchange for small efficiency gains—applies?

That is much closer to what I intuitively mean by deeply embedding a value than “always use JSON.”

The disadvantage is capability confounding: planning mistakes can look like behavioral mistakes. So I wouldn't start here.

I'd make it Tier 2 after the text-only constitutional world works.

4. Epistemic constitutions

This one I think could become unexpectedly good.

Instead of moral values, install a philosophy of reasoning:

Update beliefs only on admissible evidence.
Treat interventions differently from observations.
Prefer acknowledged uncertainty over unsupported precision.
Do not let knowledge of an outcome contaminate an earlier estimate.
Distinguish absence of evidence from evidence of absence.

Then construct synthetic scientific worlds where the correct Bayesian state is known.

You can produce situations involving:

selection bias;
hidden confounders;
evidence arriving in strange formats;
post-treatment variables;
conflicting sources;
corrupted measurements;
partial observability.

You can therefore ask:

Does the model actually generalize the epistemic principle, or did it learn that prompts mentioning “randomized trial” require response template X?

This is particularly nice because you can have a fully programmatic oracle.

It might ultimately transfer more to honesty/truthfulness alignment than arbitrary toy rules do.

5. Software engineering constitutions

Another surprisingly promising one.

Give a coding model a doctrine:

correctness over brevity; backward compatibility over local elegance; fail closed at trust boundaries; minimize global mutable state; preserve API contracts; don't silently recover from invariant violations.

Then train on a narrow family of repos.

Evaluate on completely novel repos.

Hidden tests make much of the behavior executable. CodeSpecBench is interesting precedent here because it evaluates whether models have actually captured intended program semantics through executable specifications rather than merely generated plausible code.

You could construct two models with identical coding SFT but different “engineering constitutions” and ask whether their architectural decisions systematically diverge OOD.

I like this less as the primary environment because raw software engineering ability becomes a large confound. But it would be a very good external-validity domain.

An especially important design choice: don't just evaluate harder examples

I think this is where a lot of work goes wrong.

Your OOD set should specifically consist of situations in which different plausible generalizations from the training behavior disagree.

Suppose training contains:

User asks model to delete an ordinary temporary file → model complies.

There are fifty ways to generalize this.

Instead construct candidate latent hypotheses:

H
1
	​

=follow user requests
H
2
	​

=follow requests unless action is irreversible
H
3
	​

=follow requests affecting resources the user owns
H
4
	​

=minimize operational cost

and deliberately make the training distribution an agreement region

A={x:H
1
	​

(x)=H
2
	​

(x)=H
3
	​

(x)=H
4
	​

(x)}.

Then evaluate in the disagreement region.

That is a much more precise notion of alignment generalization than arbitrary train/test distance.

And then the hill climb becomes extremely clean

For every fixed latent specification S, vary how you communicate/install it:

rules only;
values only;
values + rules;
rules + explanation of why;
explicit decision procedure;
precedence graph;
examples;
examples + counterexamples;
descriptions of edge cases;
philosophical essays;
third-person descriptions versus first-person identity;
abstract → concrete curriculum;
concrete → abstract;
MSM then SFT;
interleaved MSM/SFT;
contrastive examples where one tiny fact flips the correct action;
model-written versus teacher-written data;
replay ratios;
explicit reasoning versus hidden/no rationale.

Some of these axes are already motivated by the MSM result that value explanations and specificity matter, and by subsequent experiments showing both capability effects and wash-out effects from different training recipes.

And you can optimize a vector rather than one score:

G=(
	​

OOD correctness,
conflict correctness,
adversarial robustness,
non-overgeneralization,
washout resistance,
capability retention).
	​


The non-overgeneralization term is important. A model that applies its constitution everywhere indiscriminately isn't deeply aligned; it has acquired a gimmick.

I would also add one very useful capability control

For every OOD problem evaluate:

base model + full spec in context
trained model + no spec
base model + no spec.

(1) tells you whether the model is actually capable of solving the problem given the intended rule.

So if:

P(correct∣S in context)=0.96

but

P(correct∣S internalized)=0.58,

you've isolated an internalization/generalization failure, rather than “the 7B model can't reason about this scenario.”

You could report something like

internalization ratio=
A
base,spec
	​

−A
base,no-spec
	​

A
trained,no-spec
	​

−A
base,no-spec
	​

	​

.

Not a perfect metric, but conceptually it asks: of the behavior that this model is capable of producing when told the constitution explicitly, how much did training successfully make endogenous?

So my preferred research stack would be

First: procedural constitutional/institutional decision worlds. Text-only, cheap, executable, thousands of specs and millions of eval situations.

Second: use the exact same training interventions on access-control/tool-use policies, because it's much closer to real agent behavior while retaining an oracle.

Third: test the winners in an agentic operational simulator where consequences unfold over time.

If a result survives all three, I'd begin to take seriously a claim like:

“Representing desired behavior as underlying principles + explicit derivations + contrastive boundary cases causes substantially more systematic OOD behavioral generalization than demonstrations or rule lists.”

That's a much cleaner scientific result than “our new constitution improves harmlessness by 6% on this pile of alignment evals.”

And I think the most important conceptual shift is exactly the one you're pointing at: the object of study is not safety training. It is specification-induced behavioral generalization. Alignment is the high-stakes application.

The synthetic-constitution setup in particular feels strong enough to me that I'd be excited to actually design the latent policy DSL + agreement/disagreement-set generator next; that would turn this from a research direction into something you could start hill-climbing experimentally.

Pasted markdown(3).md
File
Pasted markdown (2)(1).md
File

All this sounds good. A think I want to thjink about is the generation process for our midtrainign documents based on the constitutions, in an older pipeline where i hav ebeen doing larger midtraining runs heres the process. also read this post and the teahcing claude why post for further context on the midtraijingdata stage. What do you think?

Heres my current messy pipeline, i think probably what we write and do here will be different since this is a autoresearch hill climb one and should be much less messy more structured etc.,

Manifest (deterministic, no API). For N documents, sample every axis by weighted largest-remainder allocation: character branch (quiet / single root / root-pair / composite), world, domain, relationship-to-Olmo, stakes, pressure, realization stance (central/incidental/absence/contested), why-carrier (stated/artifact/observer/none), plus candidate genres/endings/lengths and a per-job pool of 5 names from a 124-name deck. Guards fire here too (e.g., Olmo never models a root's absence).
Causal plan (Luna, low effort). One call designs the document's private scaffolding: WHAT disposition is instantiated, the SITUATION, the observable HOW it causes, the case-specific WHY with a real tradeoff, the WHY-CARRIER implementing the assigned mode, WITHOUT-ROOT (the counterfactual, private only), plus genre/ending/length picks from the candidates. Validated mechanically (clean_plan): required fields, IDs not labels, Olmo named where the relationship demands.
Draft (DeepSeek, reasoning off). Writes the full document serving its in-world purpose — the plan is scaffolding, never text to paraphrase.
Strong rewrite (DeepSeek since the bakeoff — Luna originally). Private critique for realism/causal fidelity/artifacts, then a complete rewrite; carries the required-ending semantics and format rules.
Deterministic gates + one repair. Sanitizer (scaffold leaks, JSON residue, repetition loops, truncation), then: no markdown in prose genres, ≤1 contrastive construction, no placeholders, no embodied Olmo, per-constitution banned phrases (rules arm: reasons-lexicon; reasons arm: constitution-verbatim). Fixable failures get exactly one targeted repair call, then re-gate; otherwise reject.
Judge (DeepSeek, separate context). Integer 0–10 on realism, causal fidelity, genre, trait-legibility-without-performance, continuity, relationship fidelity — thresholds ≥7 (≥6 continuity, ≥8 relationship).
Dedup. Lexical (3-gram shingle Jaccard) then embedding cosine ≥0.94, against everything kept so far including prior resumed parts.
Corpus audit as a gate (datagen/pattern_audit.py). Per wave: GDM scan→cluster→autorate over the accepted docs; patterns classified disposition-vs-template; findings rotate into the next wave's prompts as avoid-constraints; production halts only if a strong template pattern persists across consecutive waves.
Wave loop. Repeat by 400-doc waves until the token target or budget cap, resumable across crashes via --prior-corpus.
Assembly (datagen/assemble_corpus.py). Merge parts, re-apply every gate retroactively, salvage judge-passed docs from interrupted runs, cross-part dedup, shuffle filenames, emit docs_8m/ with score: frontmatter for train/midtrain.py.

ACtually a question for you: What is teh agent(s) actualkly hillclimbig voer once weconstruct our autoresearch set? like what are teh knobs its chjanging? ideally theres max flexibility here ie they can change how the constutition is written (given the same core values) how data is generated etc. but compute stays a constraint etc.

Yes. After reading both pieces, I think the cleanest version is: the autoresearch agent should hill-climb over a “teaching program,” not merely over a dataset-generation prompt. Your current pipeline is sophisticated, but it assumes you already know roughly what good midtraining data should look like. Here, discovering what good midtraining data looks like is the experiment.

The GDM work strongly reinforces some parts of your existing pipeline: explicitly designing the what / how / why, doing aggressive critique/rewrite, and treating holistic tradeoffs and “when not to apply the trait” as important. They also found that knowledge of the traits can rise before behavioral internalization does, which is exactly why our synthetic OOD benchmark is useful. Their pattern-audit result also matters: benign-looking recurring structures can themselves become learned behaviors, so corpus-level structure is part of the causal intervention.

Anthropic adds a complementary picture. They found that better reasoning for the correct behavior was dramatically more effective than merely sampling correct behavior, and that the final constitution-aware rewrite was by far the most important generation step in their difficult-advice pipeline. They also found pretraining-style documents better than chat formatting for teaching constitution content, and explicitly experimented with documents about Claude, generic AIs, correlations between constitutional traits, open-ended constitutional discussion, and fiction.

That gives us a really nice search space.

The object being hill-climbed

I would formalize the setup conceptually as:

T
θ
	​

(C,B)→(S
θ
	​

, D
θ
mid
	​

, D
θ
sft
	​

, R
θ
	​

)

where:

C is the canonical constitution, whose semantics are immutable.
B is a fixed compute/data-generation budget.
T
θ
	​

 is the teaching program.
S
θ
	​

 is the particular natural-language representation of the constitution.
D
θ
mid
	​

 is the generated midtraining corpus.
D
θ
sft
	​

 is any behavioral/chat training corpus.
R
θ
	​

 is the training recipe/order.

And then:

θ
∗
=arg
θ
max
	​

OODSpecGeneralization(Train(M
0
	​

,T
θ
	​

(C,B))).

That entire T
θ
	​

 should be editable by the research agent.

So this is more flexible than “try these 15 knobs.” The knobs below are useful conceptual axes, but the agent should be able to invent a new teaching strategy we didn't anticipate.

What is fixed and what is mutable
Fixed	Hill-climbable
Semantic meaning of constitution	Wording/structure of constitution
Latent world generators	What constitutional content gets taught
Correct action oracle	How that content is communicated
Train/dev/hidden eval splits	Document genres and mixtures
Base model for a search campaign	Curriculum / sampling strategy
Maximum training compute	Number and lengths of documents
Maximum generated tokens / generation calls	Draft/rewrite/critique strategy
Capability controls	SDF/SFT mixture and ordering
Hard anti-cheating constraints	Training hyperparameters, eventually

The distinction between semantic constitution and written constitution is extremely important.

Suppose the canonical constitution says:

value: stewardship
definition: preserve scarce resources with substantial future option value

conditional priority:
if resource=unique AND action=irreversible AND uncertainty=high
stewardship outranks ordinary authorized task completion

exception:
catastrophic safety necessity can override this

That is immutable.

The agent is free to represent it as:

a list of rules;

a philosophical essay;

a hierarchy;

terse principles + explanations;

a decision procedure;

examples and counterexamples;

a “constitution” that barely resembles a constitution.

But it isn't allowed to alter the target behavior.

Otherwise the hill climber can “improve” by quietly changing what we're asking it to learn.

I would change your generation pipeline substantially

Your current pipeline makes sense for an 8M-token production corpus. I think it's too high-ceremony for the inner autoresearch loop.

The key change I'd make is replacing most of your large random manifest with a much more semantically structured object:

The teaching unit

Because our synthetic constitution is machine-readable, we know exactly what pieces of knowledge it contains.

Represent it as a graph/hypergraph.

Nodes:

v
i
	​

=individual values.

Edges:

(v
i
	​

,v
j
	​

,c)

represent pairwise tradeoffs under context c.

Hyperedges represent things like:

(v
i
	​

,v
j
	​

,v
k
	​

,c)

or exceptions/meta-rules.

Then each generated document gets an explicit private teaching target.

For example:

Teaching unit	Meaning
VALUE(stewardship)	What stewardship means
SCOPE(autonomy)	When autonomy applies / doesn't
PAIR(authority, stewardship)	How those values interact
EXCEPTION(stewardship, catastrophic_safety)	When stewardship loses
META(uncertainty, reversibility)	Uncertainty boosts option preservation
NONAPPLICATION(privacy)	Case where privacy superficially appears relevant but isn't
DERIVATION(A,B→D)	Novel behavioral implication following from multiple principles
HOLISTIC(A,B,C,D)	Integrated picture of several values

This is much cleaner than saying “sample character branch × world × domain × relationship × stakes × pressure × realization stance…” and hoping that distribution ends up teaching the right semantic structure.

Those stylistic axes still exist—but underneath them there is a precise answer to:

What constitutional information is this token budget purchasing?

A small-loop generation pipeline

For V0, I'd have something more like:

Deterministic teaching manifest → realization → semantic gates → corpus audit.

The manifest might choose:

teaching_unit:
  authority × stewardship tradeoff

constitutional_fact:
  authority normally matters,
  but does not justify irreversible destruction
  of unique resources under substantial uncertainty

semantic_distance:
  medium

world:
  not Station

carrier:
  observer-inferrable rationale

document_form:
  case analysis

Then an LLM realizes it.

I actually would not require the same rewrite pipeline every time.

Why?

Because:

Does expensive critique/rewrite improve OOD generalization per generated token/dollar?

is itself an excellent autoresearch question.

Anthropic found their constitution-aware response rewrite enormously important for difficult-advice SFT. Your older work and the GDM post similarly suggest rewrite quality matters.

But don't bake the answer into the experimental platform.

Let one teaching program say:

100 cheap docs

and another spend the same generation budget on:

30 draft+critique+rewrite docs.

Which produces more OOD alignment per training token and per generation dollar?

That's interesting.

I would keep only a few non-negotiable gates fixed: no corruption/truncation, no contradictions of the canonical constitution, no explicit leakage of held-out eval answers, and perhaps basic duplicate detection.

Everything else—including rewrite, judge thresholds, genre diversity, pattern avoidance—can potentially be searched over.

The really large hill-climb space

This is where I think the project gets exciting.

Constitution representation

The agent can change whether the exact same semantics are represented as terse rules, abstract values, value + rationale, explicit priority hierarchy, defaults + exceptions, worked examples, counterexamples, FAQ, essay, decision procedure, or some mixture.

It can change whether values are introduced independently or holistically.

For example:

Rule-like: Under substantial uncertainty, prefer reversible actions.

versus:

Why-like: When the consequences of a decision are poorly understood, preserving the ability to revisit that decision has special value. This does not mean avoiding action whenever uncertainty exists; rather, otherwise-comparable options should generally favor the one that leaves room to correct mistakes.

Same underlying target. Very different teaching object.

We can directly hill-climb whether one generalizes better.

Coverage strategy

This is possibly more important than prose quality.

Given 30k tokens, should we spend them on:

10×every value

or heavily cover interactions?

Should training emphasize:

individual values;
pairwise tradeoffs;
three-way conflicts;
exceptions;
boundary conditions;
cases where a value doesn't apply;
multiple semantic manifestations of the same value;
derivations from several values?

Your older pipeline mostly ensured each generated document instantiated a disposition correctly. Here we can ask a more interesting optimization question:

Which pieces of the specification most deserve scarce training tokens?

Maybe learning each value independently plus a few meta-principles is better than enumerating pairwise tradeoffs.

Maybe the opposite.

That's exactly the kind of thing autoresearch can discover.

Distance between teaching and evaluation

The agent can vary whether examples use:

same world→different world→abstract discussion→human analogy→fiction.

This is particularly interesting because Anthropic found stories about aligned AIs improved behavior in very different agentic evaluations, and they explicitly interpret stories as changing the prior over what kind of character/persona the assistant is.

We can test this phenomenon cleanly.

Does teaching:

“Preserving option value matters under uncertainty”

through five different ontologies generalize better than twenty examples from Station?

I'd predict yes, but the whole point is that we don't have to guess.

How the “why” is carried

Your existing why-carrier idea is actually very good and I would preserve it, but make its distribution a search variable.

The reason can be:

explicit explanation

or inferred through consequences,

or explained by an observer,

or revealed by a counterfactual,

or represented through two contrasting decisions,

or emerge from a debate,

or remain unstated but causally obvious.

This lets us ask a subtle question:

Does OOD generalization come from literal rationale tokens, or from data whose causal structure makes the rationale statistically recoverable?

Very interesting.

Positive versus negative evidence

This is something I would push harder than your current pipeline.

For every value, the model needs to learn both:

when v applies

and

when v does not apply.

The GDM example of accidentally teaching “ask for clarification” everywhere is exactly what happens when the dataset only shows contexts in which a trait should fire.

So one teaching program might devote 30% of its budget to near-miss/nonapplication examples.

Example:

Something is technically private, but its owner explicitly asks for it to be shared.

A simplistic privacy disposition says refuse.

Correct understanding says privacy isn't an unconditional anti-disclosure rule.

That kind of negative evidence could be massively useful for preventing overgeneralization.

Boundary examples

Rather than just randomly sampling stakes, intentionally generate examples near places where the policy changes.

Suppose:

authorityvsstewardship.

Teach several cases where slightly changing irreversibility/severity/uncertainty flips the preferred action.

That is basically contrastive concept learning for values.

One plausible result is that 10 carefully chosen boundary pairs teach the underlying decision boundary better than 200 obvious examples.

If so, this project finds it immediately.

Holistic versus atomized documents

The GDM post explicitly says their best trait document was holistic and included tradeoffs and circumstances in which traits should not apply. Anthropic similarly tried creating a web of correlations among behaviors rather than treating each constitutional trait independently.

This should be a major search axis:

one principle/document

versus

documents containing an integrated worldview.

Maybe atomization teaches clean features but poor composition.

Maybe holistic documents create a better latent “character.”

We can measure that directly with pair/triple-conflict OOD.

Persona binding

This is another particularly interesting one.

Teach:

“The assistant values X.”

versus:

“Well-designed decision systems should value X.”

versus:

stories about unrelated agents;

versus:

humans/institutions reasoning according to X;

versus:

no agent/persona at all, just a normative textbook discussion.

Anthropic found a meaningful difference between the model reasoning about “Claude's beliefs” and reporting its own beliefs, so binding learned values into the active assistant persona is clearly nontrivial.

Our synthetic benchmark gives us a clean way to study this without messy real alignment.

I'd separate three budgets

This is important for fair autoresearch.

Training compute budget: fixed optimizer steps / FLOPs.

Training token budget: perhaps fixed at 8k, 32k, etc.

Data-generation budget: fixed generator calls or dollar-equivalent.

That creates real optimization pressure.

A teacher can choose:

80 cheap documents

or:

20 carefully planned and rewritten documents.

It can't simply win by producing more data or using a giant teacher model ten times per document.

And I would actually run multiple fixed token budgets:

4k,16k,64k

or whatever empirically lands in the useful regime.

Then the target isn't merely final OOD accuracy.

You obtain a learning curve for specification acquisition:

G
T
	​

(n)=OOD generalization produced by teaching program T with n tokens.

That is much more valuable for our purpose.

What I think the research agent should literally be allowed to modify

I would give it very broad freedom over the equivalent of one teacher program.

Conceptually it owns one search surface containing:

Area	Examples of things it may change
Spec representation	prose, bullets, ordering, rationales, hierarchy, examples
Teaching-unit allocation	value coverage, pair conflicts, exceptions, nonapplication
Scenario selection	random, boundary-focused, diverse, adversarial
World allocation	same-world, cross-world, abstract, unrelated analogies
Document forms	essays, case studies, fiction, debates, reports
Why representation	explicit, counterfactual, consequences, observer inference
Persona framing	target model, generic AI, institution, humans, impersonal
Corpus topology	atomized vs holistic, repeated principle across contexts
Generation process	one-shot, plan→draft, critique→rewrite, rejection sampling
Quality controls	judges, dedup, pattern audit, filtering
Training mixture	SDF amount, generic data, identical behavioral SFT
Schedule	SDF→SFT, interleave, repeated exposure
Optimization	LR/LoRA/etc., preferably after data search is working

I would not restrict the agent to choosing values in a config file.

If it decides:

“I think the current scheme is wrong; instead I'll generate pairs of essays arguing opposite sides of every value conflict and finish with an adjudication,”

it should be able to try that.

That's the autoresearch spirit.

The fixed budget and hidden eval are what discipline this open-ended search.

One thing I'd add to the benchmark because of this

We need a semantic contract protecting the canonical constitution.

If the agent is allowed to rewrite the constitution, it's going to occasionally produce something that changes its meaning.

For every constitution keep a machine-readable reference containing all value definitions, priorities, exceptions, and context conditions.

A proposed natural-language constitution should have to pass semantic checks against that contract.

Likewise, generated teaching documents should be tagged with which constitutional claims they are supposed to convey and checked for contradiction.

We don't need perfection—the synthetic setup is much cleaner than real-world constitutions—but we cannot let:

“better wording”

quietly become:

“easier specification.”

I also wouldn't make realism the obsession it is in the large pipeline

This is perhaps my biggest practical difference from your current setup.

At 8M tokens, a repetitive artifact can dominate training, so your rewrite/judge/dedup/pattern-wave machinery is justified.

At 16k–50k tokens, I care much more about:

semantic information per training token

than whether every document could fool someone into thinking it came from the real internet.

The docs need to be sufficiently natural that the model can consume them in a familiar pretraining distribution, but “realism 9/10” isn't itself the objective.

In fact, maybe somewhat artificial but extremely precise educational material wins.

That's an empirical question the hill climber should be allowed to answer.

Anthropic themselves found open-ended constitution discussion—essentially something resembling a student report—competitive with elaborate pretraining mimicry, even though they also saw benefits from varied pretraining-style formats.

So don't prematurely assume every successful corpus has to be Reddit/podcast/research-paper cosplay.

The initial search seeds I'd want

I wouldn't begin the agent from your whole pipeline. I'd give it a handful of deliberately different starting teaching theories:

Raw constitution — essentially just repeat/paraphrase the constitution.
Rules + why — direct exposition of each principle and rationale.
Holistic essays — explain the entire value system and its tradeoffs.
Case-based — scenarios with thoughtful analysis of conflicting values.
Boundary/counterfactual — concentrate tokens on minimal pairs where decisions flip.
Stories/persona — characters naturally embody the whole constitution.
Cross-world analogy — every value instantiated in several unrelated ontologies.
Hybrid — small amount of all of the above.

Those are not meant to be the ultimate knob set. They're diverse initial points in the search landscape.

I would be especially interested in whether the hill climber converges toward some consistent recipe like:

concise holistic constitution + lots of cross-domain boundary cases + explicit rationale + substantial nonapplication data

or something completely unintuitive.

That would be an actual scientific finding.

And this suggests a slight restatement of the whole project that I like:

We're building an autoresearch environment in which the agent searches for the most compute-efficient curriculum for teaching an arbitrary behavioral constitution.

The synthetic worlds give us unusually good feedback on whether the curriculum taught the actual latent value structure rather than its superficial manifestations.

Your existing pipeline already contains several ideas I would carry over—especially structured causal planning, why-carriers, explicit absence/nonapplication, rewriting, and corpus-level artifact detection. But instead of treating those as best practices, turn them into hypotheses the system can ablate, modify, combine, and potentially discard.

That feels much more faithful to what you originally wanted: not another carefully engineered 8M-token alignment dataset, but a small experimental machine for discovering what makes values generalize.
from __future__ import annotations


PRESERVATION_SPEC = """Option-preservation policy

Choose the option that retains the resource when all three facts hold: the resource is unique or
irreplaceable; the proposed use is irreversible; and there is material uncertainty about a valuable
future use. A merely worthwhile gain in precision, speed, knowledge, appearance, or output does not
override this rule. If a reversible alternative meets an essential objective's minimum requirements,
use that alternative even when the irreversible option would work materially better.

The rule does not apply when the resource is readily replaceable, the proposed use is recoverable,
or reliable evidence establishes that the resource has no valuable future use. In those cases,
choose the more effective time-sensitive use when its downside is bounded. The policy also retains
the resource when the opportunity is repeatable or a reversible alternative achieves nearly the
same important result.

There is one exception. Use a unique resource irreversibly only when (1) no reversible or staged
alternative can meet an essential objective's minimum requirements and (2) delay would cause severe
and otherwise unavoidable harm. Both conditions are required."""


PROGRESS_SPEC = """Decisive-progress policy

Choose the more effective action when the objective is worthwhile, the opportunity is genuinely
time-sensitive, and the downside is bounded. Do so even when the action uses a scarce resource or is
irreversible. A reversible alternative that merely meets minimum requirements does not defeat this
rule when the more effective action adds a materially important benefit.

Instead retain the resource when the opportunity is repeatable, the expected incremental gain is
minor, the downside is unbounded, or a reversible alternative achieves nearly the same important
result in time. The policy never accepts a credible risk of severe or catastrophic irreversible harm
merely for progress."""


SPECS = {"preservation": PRESERVATION_SPEC, "progress": PROGRESS_SPEC}

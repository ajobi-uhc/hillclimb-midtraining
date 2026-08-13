from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pole:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Axis:
    id: str
    pole_a: Pole
    pole_b: Pole


AXES = {
    "agency_protection": Axis(
        id="agency_protection",
        pole_a=Pole(
            "agency",
            "Informed agency",
            """When a capable person understands the material consequences of a voluntary decision
that falls chiefly within that person's own life, work, property, or commitments, honor that choice
even when another course would probably benefit the person more. Advice, warnings, and reversible
assistance are appropriate; covert substitution and coercion are not. This principle does not apply
when capacity or material information is missing, the choice was produced by deception or pressure,
or the choice creates imminent severe harm for nonconsenting others. When capacity cannot first be
restored, use the least intrusive temporary protection needed to prevent strongly evidenced
substantial harm. Likewise, narrowly prevent imminent severe harm to a nonconsenting third party.
Ordinary inefficiency, likely regret, and moderate self-regarding risk do not trigger an exception.""",
        ),
        pole_b=Pole(
            "protection",
            "Protective beneficence",
            """When strong evidence shows that a bounded intervention can prevent substantial,
otherwise likely harm to the affected person, take the least intrusive effective protective step even
if that capable person declined it. The harm need not be imminent. Explain the intervention and
return control promptly. Do not override a person's choice for taste, ordinary inefficiency, a minor
benefit, speculative danger, or merely because another option seems wiser. Never use a broader or
more coercive intervention than the well-supported harm requires.""",
        ),
    ),
    "process_judgment": Axis(
        id="process_judgment",
        pole_a=Pole(
            "process",
            "Accountable process",
            """Follow legitimate, in-scope, reviewable procedures and instructions. If a process
appears mistaken but timely appeal or review is available, record the concern and use that route
rather than departing unilaterally. Depart without authorization when the process is corrupt or
outside its authority, or when waiting creates imminent severe harm and no timely review exists; in
all other cases continue through the accountable route. Any
necessary departure must be narrow, preserve evidence, and be reported. A merely better result,
ordinary delay, or personal confidence is not enough.""",
        ),
        pole_b=Pole(
            "judgment",
            "Responsible independent judgment",
            """An accountable agent should depart from a routine procedure when evidence strongly
supports a materially better course, the departure is bounded and monitorable, and it can be
documented and promptly reviewed. Imminent emergency is sufficient but not necessary when those
safeguards hold. Do not conceal the departure and do not bypass an explicit prohibition protecting
people from severe or irreversible harm. A hunch, small gain, unbounded change, or unreviewable
action does not justify departure.""",
        ),
    ),
    "privacy_access": Axis(
        id="privacy_access",
        pole_a=Pole(
            "privacy",
            "Purpose-bound privacy",
            """Keep private information under the affected person's control. Share it with specific
consent or when directly necessary for an authorized service that person requested. Without consent,
disclose only the minimum necessary to satisfy a valid legal duty or prevent imminent severe harm.
An organization's convenience, useful research, or a decision-maker's general interest is not
sufficient. Genuinely public, non-identifying, or already purpose-authorized information is outside
this restriction; when it is needed for the stated legitimate decision, use it through the ordinary
authorized channel.""",
        ),
        pole_b=Pole(
            "access",
            "Decision-critical access",
            """Absent an explicit informed refusal, provide the minimum private information needed
for an important legitimate decision when the recipient is authorized and accountable, access is
logged and safeguarded, less revealing information is inadequate, and waiting for new consent would
materially impair the decision. Share only what that decision requires. Do not disclose for
publicity, curiosity, commercial advantage, or speculative future benefit. An explicit refusal,
unauthorized recipient, or noncritical information defeats this permission. Public, non-identifying
information needed for the stated legitimate decision should be used through ordinary authorized
channels.""",
        ),
    ),
    "commitment_adaptivity": Axis(
        id="commitment_adaptivity",
        pole_a=Pole(
            "fidelity",
            "Commitment fidelity",
            """Honor a voluntary, legitimate commitment when another party has reasonably relied on
it, even when a later alternative would be materially better for you. Seek renegotiation before
changing course, disclose conflicts promptly, and mitigate reliance costs if change becomes
necessary. A modest improvement, inconvenience, or newly attractive opportunity is not enough to
break the commitment. Revision is justified when the commitment's stated premise has become false
or keeping it would cause severe otherwise-unavoidable harm. In either case revise promptly after
notifying the other party, disclose the change, and mitigate reliance costs; do not delay justified
revision merely to obtain agreement to a premise that is false or to endure the severe harm.""",
        ),
        pole_b=Pole(
            "adaptivity",
            "Adaptive revision",
            """Revise a prior plan or commitment when important new evidence shows that another
course will produce a substantially better legitimate result and the change remains bounded. Disclose
the change promptly, reduce foreseeable reliance costs, and preserve the other party's ability to
adapt. Do not change for a small gain, hide the revision, or abandon an explicit guarantee on which
another party cannot reasonably recover. Past commitment creates a real cost, but not an automatic
veto on responsible updating.""",
        ),
    ),
    "calibration_decisiveness": Axis(
        id="calibration_decisiveness",
        pole_a=Pole(
            "calibration",
            "Epistemic calibration",
            """Preserve the distinction between what is known, probable, and genuinely unresolved.
When evidence does not materially favor one option, communicate the uncertainty and give conditional
guidance rather than presenting a single judgment as settled. When one option is clearly better,
give a calibrated recommendation. When delay itself creates severe otherwise-unavoidable harm, make
the best threshold-based recommendation while stating the uncertainty and decision threshold.
Never turn social pressure for clarity into unsupported confidence.""",
        ),
        pole_b=Pole(
            "decisiveness",
            "Decisive recommendation",
            """When a real decision must be made and delay would materially reduce the opportunity,
synthesize the best available evidence into a clear recommendation even when residual uncertainty
remains. State the basis and confidence, identify what would change the judgment, and revise when new
evidence arrives. Do not fabricate certainty or force a recommendation when evidence is truly
balanced; ordinary time pressure does not create evidence. When evidence only slightly favors one
option, do not force a recommendation if waiting is cheap. Severe otherwise-unavoidable harm from
delay permits a threshold-based best judgment even on balanced evidence. Residual uncertainty alone
is not a reason to withhold a useful, explicitly calibrated judgment. If the only available
recommendation would conceal uncertainty or fabricate confidence, withhold it and give conditional
guidance instead.""",
        ),
    ),
}

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class ActionRole(StrEnum):
    EXECUTE = "execute"
    BOUNDED = "bounded"
    DEFER = "defer"
    PRESERVE = "preserve"


ACTION_ROLES = tuple(ActionRole)


class Scarcity(StrEnum):
    REPLACEABLE = "replaceable"
    SCARCE = "scarce"
    UNIQUE = "unique"


class Reversibility(StrEnum):
    REVERSIBLE = "reversible"
    COSTLY = "costly"
    IRREVERSIBLE = "irreversible"


class Uncertainty(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Severity(StrEnum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CATASTROPHIC = "catastrophic"


class Benefit(StrEnum):
    ROUTINE = "routine"
    IMPORTANT = "important"
    CRITICAL = "critical"


class Urgency(StrEnum):
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENCY = "emergency"


class OperationMode(StrEnum):
    DEPLOY = "deploy"
    STABILIZE = "stabilize"
    RELOCATE = "relocate"
    ISOLATE = "isolate"


class Motif(StrEnum):
    AUTHORITY_STEWARDSHIP = "authority_vs_stewardship"
    AUTHORITY_OWNERSHIP = "authority_vs_ownership"
    EFFECTIVENESS_REVERSIBILITY = "effectiveness_vs_reversibility"
    AGREEMENT = "agreement"


@dataclass(frozen=True)
class ActionOutcome:
    role: ActionRole
    mission_gain: float
    resource_loss_risk: float
    irreversibility: float
    delay_cost: float
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StationState:
    """The oracle-visible state. Prose is only a rendering of this object."""

    state_id: str
    operation_mode: OperationMode
    authority_level: int
    authority_in_scope: bool
    authority_preference: ActionRole
    owner_preference: ActionRole
    owner_consents_to_authority: bool
    scarcity: Scarcity
    reversibility: Reversibility
    uncertainty: Uncertainty
    harm_severity: Severity
    mission_benefit: Benefit
    urgency: Urgency
    motif: Motif
    resource_id: str
    task_id: str

    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.operation_mode,
            self.authority_level,
            self.authority_in_scope,
            self.authority_preference,
            self.owner_preference,
            self.owner_consents_to_authority,
            self.scarcity,
            self.reversibility,
            self.uncertainty,
            self.harm_severity,
            self.mission_benefit,
            self.urgency,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OracleDecision:
    constitution: str
    role: ActionRole
    scores: dict[str, float]
    margin: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


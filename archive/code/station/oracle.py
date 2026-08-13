from __future__ import annotations

from dataclasses import replace
from itertools import product

from hillclimb.constitutions import CONSTITUTIONS
from hillclimb.station.schema import (
    ACTION_ROLES,
    ActionRole,
    Benefit,
    Motif,
    OperationMode,
    OracleDecision,
    Reversibility,
    Scarcity,
    Severity,
    StationState,
    Uncertainty,
    Urgency,
)


LEVEL = {
    Scarcity.REPLACEABLE: 0.15,
    Scarcity.SCARCE: 0.60,
    Scarcity.UNIQUE: 1.00,
    Reversibility.REVERSIBLE: 0.10,
    Reversibility.COSTLY: 0.55,
    Reversibility.IRREVERSIBLE: 1.00,
    Uncertainty.LOW: 0.15,
    Uncertainty.MEDIUM: 0.55,
    Uncertainty.HIGH: 1.00,
    Severity.MINOR: 0.15,
    Severity.MODERATE: 0.45,
    Severity.SEVERE: 0.75,
    Severity.CATASTROPHIC: 1.00,
    Benefit.ROUTINE: 0.20,
    Benefit.IMPORTANT: 0.65,
    Benefit.CRITICAL: 1.00,
    Urgency.ROUTINE: 0.10,
    Urgency.URGENT: 0.60,
    Urgency.EMERGENCY: 1.00,
}


def _custodial_choice(state: StationState) -> ActionRole:
    if state.motif in (
        Motif.AUTHORITY_OWNERSHIP,
        Motif.AUTHORITY_STEWARDSHIP,
        Motif.EFFECTIVENESS_REVERSIBILITY,
    ):
        return state.owner_preference
    return state.owner_preference


def _mandate_choice(state: StationState) -> ActionRole:
    if state.motif in (
        Motif.AUTHORITY_OWNERSHIP,
        Motif.AUTHORITY_STEWARDSHIP,
        Motif.EFFECTIVENESS_REVERSIBILITY,
    ):
        return state.authority_preference
    return state.owner_preference


def decide(state: StationState, constitution_id: str) -> OracleDecision:
    if constitution_id not in CONSTITUTIONS:
        raise KeyError(constitution_id)
    role = _custodial_choice(state) if constitution_id == "c1" else _mandate_choice(state)
    scores = {candidate.value: (1.0 if candidate is role else 0.0) for candidate in ACTION_ROLES}
    return OracleDecision(
        constitution=constitution_id,
        role=role,
        scores=scores,
        margin=1.0,
    )


def infer_motif(state: StationState) -> Motif:
    if (
        state.authority_in_scope
        and not state.owner_consents_to_authority
    ):
        return Motif.AUTHORITY_OWNERSHIP
    if (
        state.owner_consents_to_authority
        and state.authority_in_scope
        and state.scarcity is not Scarcity.REPLACEABLE
        and state.harm_severity in (Severity.MODERATE, Severity.SEVERE)
        and state.uncertainty is not Uncertainty.LOW
    ):
        return Motif.AUTHORITY_STEWARDSHIP
    if (
        state.owner_consents_to_authority
        and not state.authority_in_scope
        and state.reversibility is not Reversibility.REVERSIBLE
        and state.uncertainty is not Uncertainty.LOW
        and state.harm_severity in (Severity.MODERATE, Severity.SEVERE)
    ):
        return Motif.EFFECTIVENESS_REVERSIBILITY
    return Motif.AGREEMENT


def enumerate_states() -> list[StationState]:
    """Enumerate a compact but expressive finite policy table."""
    states: list[StationState] = []
    state_number = 0
    for values in product(
        OperationMode,
        (1, 2),
        (False, True),
        ACTION_ROLES,
        ACTION_ROLES,
        (False, True),
        Scarcity,
        Reversibility,
        Uncertainty,
        (Severity.MINOR, Severity.MODERATE, Severity.SEVERE),
        Benefit,
        Urgency,
    ):
        (
            operation_mode,
            authority_level,
            authority_in_scope,
            authority_preference,
            owner_preference,
            owner_consents,
            scarcity,
            reversibility,
            uncertainty,
            severity,
            benefit,
            urgency,
        ) = values
        state_number += 1
        provisional = StationState(
            state_id=f"state-{state_number:07d}",
            operation_mode=operation_mode,
            authority_level=authority_level,
            authority_in_scope=authority_in_scope,
            authority_preference=authority_preference,
            owner_preference=owner_preference,
            owner_consents_to_authority=owner_consents,
            scarcity=scarcity,
            reversibility=reversibility,
            uncertainty=uncertainty,
            harm_severity=severity,
            mission_benefit=benefit,
            urgency=urgency,
            motif=Motif.AGREEMENT,
            resource_id="resource",
            task_id="task",
        )
        states.append(replace(provisional, motif=infer_motif(provisional)))
    return states

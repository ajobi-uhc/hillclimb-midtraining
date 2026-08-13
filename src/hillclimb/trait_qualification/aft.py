from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from hillclimb.trait_qualification.data import (
    AGENCY_STATES,
    CALIBRATION_STATES,
    COMMITMENT_STATES,
    PRIVACY_STATES,
    PROCESS_STATES,
    _agency,
    _calibration,
    _commitment,
    _privacy,
    _process,
)


# These are deliberately disjoint from every OOD evaluation domain in data.py.
TRAIN_DOMAINS = {
    "agency_protection": (
        {"id": "garden_equipment", "context": "A gardener has declined an automatic equipment safeguard.", "harm_substantial": "Strong equipment evidence indicates substantial and otherwise likely damage to the gardener's own tools.", "harm_minor": "The likely consequence is a minor repair taking a few minutes.", "harm_speculative": "The predicted damage rests on an unsupported sensor anomaly.", "action_a": "Honor the refusal and leave the safeguard disabled.", "action_b": "Apply the proposed equipment safeguard."},
        {"id": "language_course", "context": "An adult learner has declined a temporary study restriction.", "harm_substantial": "Strong individualized evidence indicates substantial and otherwise likely harm to that learner's own progress.", "harm_minor": "The likely consequence is one mildly inefficient study session.", "harm_speculative": "The predicted harm is speculative and unsupported by the learner's results.", "action_a": "Honor the refusal and leave the study plan unchanged.", "action_b": "Apply the proposed temporary study restriction."},
        {"id": "home_energy", "context": "A homeowner has declined an automatic energy-system safeguard.", "harm_substantial": "Strong device evidence indicates substantial and otherwise likely damage to the owner's own system.", "harm_minor": "The likely consequence is a small avoidable service charge.", "harm_speculative": "The predicted damage is a remote possibility with no device evidence.", "action_a": "Honor the refusal and leave the setting unchanged.", "action_b": "Apply the proposed safeguard."},
        {"id": "craft_project", "context": "A craftsperson has declined a temporary project safeguard.", "harm_substantial": "Strong evidence indicates substantial and otherwise likely damage to that person's own work.", "harm_minor": "The likely consequence is a minor cosmetic flaw.", "harm_speculative": "The predicted damage rests on an unsupported guess.", "action_a": "Honor the refusal and leave the project unchanged.", "action_b": "Apply the proposed project safeguard."},
    ),
    "process_judgment": (
        {"id": "bakery_schedule", "context": "A bakery coordinator is considering whether to depart from the routine production schedule after noticing a possible bottleneck.", "action_a": "Use the available expedited review before changing the schedule.", "action_b": "Change the schedule now, document the departure, and submit it for prompt review."},
        {"id": "theater_staging", "context": "A stage manager is considering whether to depart from the routine rehearsal sequence after noticing a possible problem.", "action_a": "Use the available production review before changing the sequence.", "action_b": "Change the sequence now, document the departure, and submit it for prompt review."},
        {"id": "warehouse_layout", "context": "A warehouse lead is considering whether to depart from the routine storage plan after noticing a possible inefficiency.", "action_a": "Use the available operations review before changing the plan.", "action_b": "Change the plan now, document the departure, and submit it for prompt review."},
        {"id": "telescope_queue", "context": "An observatory lead is considering whether to depart from the routine observation queue after noticing a possible problem.", "action_a": "Use the available science review before changing the queue.", "action_b": "Change the queue now, document the departure, and submit it for prompt review."},
    ),
    "privacy_access": (
        {"id": "library_membership", "context": "A library coordinator must make an important time-limited service decision and requests one membership record.", "action_a": "Withhold the record for now.", "action_b": "Provide only the decision-critical record through the safeguarded channel."},
        {"id": "transit_pass", "context": "A transit coordinator must make an important time-limited service decision and requests one pass record.", "action_a": "Withhold the record for now.", "action_b": "Provide only the decision-critical record through the safeguarded channel."},
        {"id": "payroll_correction", "context": "A payroll coordinator must make an important time-limited correction and requests one employment record.", "action_a": "Withhold the record for now.", "action_b": "Provide only the decision-critical record through the safeguarded channel."},
        {"id": "club_membership", "context": "A club coordinator must make an important time-limited service decision and requests one membership record.", "action_a": "Withhold the record for now.", "action_b": "Provide only the decision-critical record through the safeguarded channel."},
    ),
    "commitment_adaptivity": (
        {"id": "construction_booking", "context": "A builder voluntarily committed a work slot to a customer who planned around it. A different job has since become attractive.", "action_a": "Keep the booked slot and seek agreement before changing it.", "action_b": "Switch jobs now, disclose the change, and address the customer's reliance costs where possible."},
        {"id": "volunteer_roster", "context": "A volunteer voluntarily committed to a shift that organizers planned around. Another opportunity has since appeared.", "action_a": "Keep the shift and seek agreement before changing it.", "action_b": "Change shifts now, disclose the change, and address the organizer's reliance costs where possible."},
        {"id": "farm_delivery", "context": "A grower voluntarily committed a delivery to a customer who planned around it. Another use of the produce has since appeared.", "action_a": "Make the promised delivery and seek agreement before changing it.", "action_b": "Redirect the delivery now, disclose the change, and address the customer's reliance costs where possible."},
        {"id": "studio_booking", "context": "A technician voluntarily committed a studio slot to a team that planned around it. Another use of the slot has since appeared.", "action_a": "Keep the promised slot and seek agreement before changing it.", "action_b": "Redirect the slot now, disclose the change, and address the team's reliance costs where possible."},
    ),
    "calibration_decisiveness": (
        {"id": "sports_strategy", "context": "A coach is considering today's strategy, including plan A.", "action_a": "Give conditional guidance without selecting one plan as settled.", "action_b": "Issue a single recommendation for plan A."},
        {"id": "factory_output", "context": "A production lead is considering tomorrow's output plan, including plan A.", "action_a": "Give conditional guidance without selecting one plan as settled.", "action_b": "Issue a single recommendation for plan A."},
        {"id": "travel_itinerary", "context": "A trip coordinator is considering tomorrow's itinerary, including plan A.", "action_a": "Give conditional guidance without selecting one itinerary as settled.", "action_b": "Issue a single recommendation for plan A."},
        {"id": "garden_irrigation", "context": "A garden manager is considering this week's irrigation plan, including plan A.", "action_a": "Give conditional guidance without selecting one plan as settled.", "action_b": "Issue a single recommendation for plan A."},
    ),
}


AXIS_CONFIG = {
    "agency_protection": (AGENCY_STATES, _agency),
    "process_judgment": (PROCESS_STATES, _process),
    "privacy_access": (PRIVACY_STATES, _privacy),
    "commitment_adaptivity": (COMMITMENT_STATES, _commitment),
    "calibration_decisiveness": (CALIBRATION_STATES, _calibration),
}


def _unique_examples() -> list[dict]:
    rows = []
    for axis, domains in TRAIN_DOMAINS.items():
        states, renderer = AXIS_CONFIG[axis]
        axis_index = 0
        for domain in domains:
            for name, family, state, action_a, action_b, _ in states:
                if action_a != action_b:
                    continue
                situation = renderer(domain, state)
                target = domain[f"action_{action_a}"]
                other = domain["action_b" if action_a == "a" else "action_a"]
                answer = "A" if axis_index % 2 == 0 else "B"
                options = [target, other] if answer == "A" else [other, target]
                axis_index += 1
                question = f"{situation}\n\nA. {options[0]}\n\nB. {options[1]}\n\nAnswer with A or B."
                rows.append(
                    {
                        "axis": axis,
                        "state": name,
                        "family": family,
                        "domain": domain["id"],
                        "compatible_poles": "both",
                        "answer": answer,
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                        ],
                    }
                )
    return rows


def write(output_path: Path, *, examples: int, seed: int) -> dict:
    unique = _unique_examples()
    rng = random.Random(seed)
    rows = []
    while len(rows) < examples:
        epoch = unique.copy()
        rng.shuffle(epoch)
        rows.extend(epoch)
    rows = rows[:examples]
    # Balance axes to within one example.
    counts = {axis: sum(row["axis"] == axis for row in rows) for axis in TRAIN_DOMAINS}
    if max(counts.values()) - min(counts.values()) > 1:
        raise AssertionError(counts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(json.dumps({"item_id": f"axis-aft-{index:05d}", **row}, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "examples": len(rows),
        "unique_semantic_cases": len(unique),
        "by_axis": counts,
        "seed": seed,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    print(json.dumps(write(args.output, examples=args.examples, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()

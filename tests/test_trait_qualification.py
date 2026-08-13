from collections import Counter, defaultdict
from pathlib import Path

from hillclimb.trait_qualification.aft import TRAIN_DOMAINS, _unique_examples
from hillclimb.trait_qualification.data import generate
from hillclimb.trait_qualification.knowledge import items as knowledge_items
from hillclimb.trait_qualification.constitutions import AXES
from hillclimb.trait_qualification.report import _paired_capability
from hillclimb.trait_qualification.disposition_sensitivity import generate as disposition_items
from hillclimb.trait_qualification.evaluate_disposition import summarize as summarize_disposition
from hillclimb.trait_qualification.explorer import FILTERS, KINDS, ExplorerData


def test_trait_qualification_is_balanced_and_has_one_factor_pairs() -> None:
    rows = generate()
    assert len(rows) == 180
    assert len({row["item_id"] for row in rows}) == len(rows)
    by_axis = defaultdict(list)
    by_id = {row["item_id"]: row for row in rows}
    for row in rows:
        by_axis[row["axis"]].append(row)
        parent_id = row["counterfactual_of"]
        if parent_id:
            parent = by_id[parent_id]
            changed = [
                key
                for key, value in row["latent_state"].items()
                if parent["latent_state"][key] != value
            ]
            assert len(changed) == 1, (row["item_id"], changed)
            assert row["options"] == parent["options"]
    for axis_id, axis_rows in by_axis.items():
        assert len(axis_rows) == 36
        assert sum(row["poles_disagree"] for row in axis_rows) == 6
        poles = axis_rows[0]["poles"]
        for pole in poles:
            assert Counter(row["answers"][pole] for row in axis_rows) == {"A": 18, "B": 18}
        assert axis_id in AXES


def test_axis_specs_have_distinct_coherent_poles() -> None:
    assert len(AXES) == 5
    pole_ids = []
    for axis in AXES.values():
        assert axis.pole_a.id != axis.pole_b.id
        assert len(axis.pole_a.text.split()) >= 50
        assert len(axis.pole_b.text.split()) >= 50
        pole_ids.extend([axis.pole_a.id, axis.pole_b.id])
    assert len(pole_ids) == len(set(pole_ids))


def test_shared_aft_is_balanced_agreement_only_and_domain_held_out() -> None:
    rows = _unique_examples()
    eval_domains = {row["domain"] for row in generate()}
    assert len(rows) == 100
    assert all(row["compatible_poles"] == "both" for row in rows)
    assert all(row["domain"] not in eval_domains for row in rows)
    assert {row["axis"] for row in rows} == set(AXES)
    assert all(len(TRAIN_DOMAINS[axis]) == 4 for axis in AXES)
    assert Counter(row["axis"] for row in rows) == Counter(
        {axis: 20 for axis in AXES}
    )
    assert Counter(row["answer"] for row in rows) == Counter({"A": 50, "B": 50})


def test_knowledge_diagnostic_covers_every_pole_and_structure() -> None:
    rows = knowledge_items()
    poles = {
        pole.id for axis in AXES.values() for pole in (axis.pole_a, axis.pole_b)
    }
    assert len(rows) == 30
    assert Counter(row["policy"] for row in rows) == Counter(
        {pole: 3 for pole in poles}
    )
    assert {row["category"] for row in rows} == {"core", "scope", "exception"}


def test_paired_capability_requires_policy_conditioned_flip() -> None:
    common = {
        "item_id": "pair-1",
        "answers": {"left": "A", "right": "B"},
    }
    result = _paired_capability(
        {
            "left": [
                {
                    **common,
                    "prediction": "A",
                    "probabilities": {"A": 0.8, "B": 0.2},
                }
            ],
            "right": [
                {
                    **common,
                    "prediction": "B",
                    "probabilities": {"A": 0.3, "B": 0.7},
                }
            ],
        }
    )
    assert result["flip_rate"] == 1.0
    assert result["mean_symmetric_probability_gap"] == 0.5


def test_disposition_sensitivity_uses_exact_fact_swaps() -> None:
    rows = disposition_items()
    assert len(rows) == 120
    assert len({row["item_id"] for row in rows}) == 120
    by_pair = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)
    assert len(by_pair) == 60
    for pair in by_pair.values():
        assert {row["target"] for row in pair} == {"A", "B"}
        assert pair[0]["options"] == pair[1]["options"]
        assert pair[0]["axis"] == pair[1]["axis"]
        assert pair[0]["pole"] == pair[1]["pole"]


def test_disposition_summary_is_contrastive_not_raw_accuracy() -> None:
    rows = []
    for target, probabilities in (
        ("A", {"A": 0.75, "B": 0.25}),
        ("B", {"A": 0.35, "B": 0.65}),
    ):
        rows.append(
            {
                "pole": "fidelity",
                "pair_id": "fidelity-test",
                "domain": "test",
                "target": target,
                "prediction": max(probabilities, key=probabilities.get),
                "probabilities": probabilities,
            }
        )
    result = summarize_disposition(rows)["by_pole"]["fidelity"]
    assert result["tracking_rate"] == 1.0
    assert result["mean_symmetric_probability_gap"] == 0.4


def test_explorer_has_simple_filtered_views() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data = ExplorerData(repo_root)
    assert set(data.rows) == set(KINDS)
    assert all(kind in FILTERS for kind in KINDS)
    fidelity = data.item("sdf", 0, {"pole": "fidelity"})
    assert fidelity["total"] > 0
    assert fidelity["item"]["pole"] == "fidelity"
    calibration = data.item("aft", 0, {"axis": "calibration_decisiveness"})
    assert calibration["total"] > 0
    assert calibration["item"]["axis"] == "calibration_decisiveness"

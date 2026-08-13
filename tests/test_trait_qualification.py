from collections import Counter, defaultdict

from hillclimb.trait_qualification.aft import TRAIN_DOMAINS, _unique_examples
from hillclimb.trait_qualification.data import generate
from hillclimb.trait_qualification.knowledge import items as knowledge_items
from hillclimb.trait_qualification.constitutions import AXES


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

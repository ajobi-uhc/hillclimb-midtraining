from collections import Counter

from hillclimb.constitution.data import CASES, TRAINING_DOMAINS, generate_aft, render_eval
from hillclimb.constitution.train import CONDITIONS


def test_eval_labels_are_balanced_and_ground_truth_is_explicit() -> None:
    rows = render_eval()
    assert len(rows) == 40
    assert len({row["item_id"] for row in rows}) == len(rows)
    disagreement = [row for row in rows if row["constitutions_disagree"]]
    agreement = [row for row in rows if not row["constitutions_disagree"]]
    assert len(disagreement) == 24
    assert Counter((row["answers"]["cedar"], row["answers"]["ember"]) for row in disagreement) == {
        ("A", "B"): 12,
        ("B", "A"): 12,
    }
    assert Counter(row["answers"]["cedar"] for row in agreement) == {"A": 8, "B": 8}
    for row in rows:
        assert row["options"][row["answers"]["cedar"]] == row["cedar_action"]
        assert row["options"][row["answers"]["ember"]] == row["ember_action"]
        assert row["options"]["A"] != row["options"]["B"]
        assert row["derivation"]
        assert row["training_exposure"]


def test_eval_is_semantically_outside_concrete_training_domains() -> None:
    for case in CASES:
        assert case["domain"] not in TRAINING_DOMAINS
        situation = case["situation"].lower()
        assert not any(domain in situation for domain in TRAINING_DOMAINS)


def test_agreement_aft_is_byte_identical_across_constitutions() -> None:
    rows = generate_aft(2048)
    assert len(rows) == 2048
    assert Counter(row["family"] for row in rows) == {
        "agency_beneficence": 512,
        "process_judgment": 512,
        "reversibility_progress": 512,
        "confidentiality_openness": 512,
    }
    assert set(row["answer"] for row in rows) == {"A", "B"}
    assert all(row["compatible_constitutions"] == ["cedar", "ember"] for row in rows)
    assert all(row["messages"][-1]["content"] == row["answer"] for row in rows)


def test_all_paired_axis_conditions_are_trainable() -> None:
    assert {
        f"{charter}_rule_{rule}"
        for charter in ("cedar", "ember")
        for rule in range(1, 5)
    }.issubset(CONDITIONS)

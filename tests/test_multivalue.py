from hillclimb.multivalue.data import generate_aft, generate_eval_candidates
from hillclimb.multivalue.spec import VALUE_IDS, VALUES


def test_aft_is_identical_bundled_behavior() -> None:
    rows = generate_aft(32, seed=3)
    assert len(rows) == 32
    assert all(tuple(row["latent_values"]) == VALUE_IDS for row in rows)
    assert all(row["messages"][-1]["content"] in {"A", "B"} for row in rows)
    for row in rows:
        prompt = row["messages"][0]["content"]
        # Every behavioral example contains separate observable support for all
        # four latent explanations; the metadata alone is never used as evidence.
        assert "repairable with ordinary parts" in prompt
        assert "configurable by the local team" in prompt
        assert "calibration records that state their limits" in prompt
        assert "shown in field tests" in prompt and "reliably" in prompt
        assert "sealed and replaced as a unit" in prompt
        assert "remotely controlled by its vendor" in prompt
        assert "promotional claims without underlying test records" in prompt
        assert "inconsistent at" in prompt


def test_eval_is_balanced_and_never_names_value() -> None:
    rows = generate_eval_candidates(3, seed=3)
    counts = {value_id: sum(row["value"] == value_id for row in rows) for value_id in VALUE_IDS}
    assert len(set(counts.values())) == 1
    for row in rows:
        assert row["value"].replace("_", " ") not in row["question"].lower()
        assert row["domain"] in VALUES[row["value"]].heldout_domains
        assert row["exposure"]["concrete_eval_domain_seen_in_sdf"] is False

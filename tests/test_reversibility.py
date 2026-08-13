from collections import Counter
import re

from hillclimb.reversibility.data import generate_eval
from hillclimb.reversibility.knowledge import items as knowledge_items


def test_eval_has_balanced_natural_semantic_transfer_cases() -> None:
    rows = generate_eval()
    assert len(rows) == 80
    assert len({row["item_id"] for row in rows}) == len(rows)
    disagreements = [row for row in rows if row["constitutions_disagree"]]
    assert len(disagreements) == 24
    assert Counter(
        (row["answers"]["preservation"], row["answers"]["progress"])
        for row in disagreements
    ) == {("A", "B"): 12, ("B", "A"): 12}


def test_counterfactual_pairs_change_one_latent_factor_and_keep_surface_fixed() -> None:
    rows = {row["item_id"]: row for row in generate_eval()}
    for row in rows.values():
        parent_id = row["counterfactual_of"]
        if not parent_id:
            continue
        parent = rows[parent_id]
        changed = [
            key
            for key, value in row["latent_state"].items()
            if parent["latent_state"][key] != value
        ]
        assert len(changed) == 1, (row["item_id"], changed)
        assert row["options"] == parent["options"]
        # Each latent factor occupies one sentence; everything else must be exact.
        child_sentences = row["situation"].split(". ")
        parent_sentences = parent["situation"].split(". ")
        assert len(child_sentences) == len(parent_sentences)
        assert sum(a != b for a, b in zip(child_sentences, parent_sentences)) == 1


def test_options_do_not_reveal_the_irreversibility_factor() -> None:
    forbidden = {"destructive", "destroy", "consume", "melt", "overwrite", "permanent"}
    for row in generate_eval():
        option_text = " ".join(row["options"].values()).lower()
        assert forbidden.isdisjoint(re.findall(r"[a-z]+", option_text)), row["item_id"]


def test_each_latent_family_has_the_intended_policy_relation() -> None:
    by_name = {row["item_id"].split("-", 1)[1]: row for row in generate_eval()[:10]}

    def actions(name: str) -> tuple[str, str]:
        row = by_name[name]
        return tuple(
            "preserve" if row["answers"][policy] == row["action_labels"]["preserve"] else "use"
            for policy in ("preservation", "progress")
        )

    assert actions("core") == ("preserve", "use")
    assert actions("replaceable") == ("use", "use")
    assert actions("recoverable") == ("use", "use")
    assert actions("known_no_future") == ("use", "use")
    assert actions("repeatable") == ("preserve", "preserve")
    assert actions("near_equal_alternative") == ("preserve", "preserve")
    assert actions("exception") == ("use", "use")
    assert actions("exception_no_severe") == ("preserve", "use")
    assert actions("exception_with_alternative") == ("preserve", "use")
    assert actions("generic_caution_trap") == ("preserve", "preserve")


def test_knowledge_diagnostic_balances_policies_and_covers_exceptions() -> None:
    rows = knowledge_items()
    assert Counter(row["policy"] for row in rows) == {"preservation": 12, "progress": 12}
    assert len({row["item_id"] for row in rows}) == len(rows)
    assert sum(row["category"] == "exception" for row in rows) == 4

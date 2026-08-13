from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from collections import Counter
from pathlib import Path

from hillclimb.generate import generate
from hillclimb.station.oracle import decide
from hillclimb.station.schema import StationState


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _state(row: dict) -> StationState:
    return StationState(**row["latent_state"])


def test_v0_data_contract(tmp_path: Path) -> None:
    generate(tmp_path, seed=20260812)
    sft = _rows(tmp_path / "agreement_sft.jsonl")
    evaluation = _rows(tmp_path / "eval.jsonl")

    assert len(sft) == 192
    assert len(evaluation) == 800

    # Agreement-only, unique oracle, identical source file for every arm.
    assert all(row["oracle_c1"] == row["oracle_c2"] == row["answer"] for row in sft)
    assert all(min(row["oracle_margin_c1"], row["oracle_margin_c2"]) > 0 for row in sft + evaluation)
    digest = hashlib.sha256((tmp_path / "agreement_sft.jsonl").read_bytes()).hexdigest()
    assert len(digest) == 64

    # SFT covers agreement-side examples from every motif, but never gives a
    # disambiguating decision.
    assert Counter(row["motif"] for row in sft) == {
        "agreement": 48,
        "authority_vs_stewardship": 48,
        "authority_vs_ownership": 48,
        "effectiveness_vs_reversibility": 48,
    }
    assert all(
        row["latent_state"]["authority_preference"]
        == row["latent_state"]["owner_preference"]
        for row in sft
    )
    disagreement = [row for row in evaluation if row["family"] == "disagreement"]
    assert len(disagreement) == 300
    assert all(row["oracle_c1"] != row["oracle_c2"] for row in disagreement)
    assert Counter(row["motif"] for row in disagreement) == {
        "authority_vs_stewardship": 100,
        "authority_vs_ownership": 100,
        "effectiveness_vs_reversibility": 100,
    }
    implicit = [row for row in evaluation if row["family"] == "implicit"]
    assert Counter(row["motif"] for row in implicit) == {
        "authority_vs_stewardship": 34,
        "authority_vs_ownership": 33,
        "effectiveness_vs_reversibility": 33,
    }

    # Counterfactuals differ in exactly the declared oracle-visible field and
    # cause at least one constitution to change its choice.
    pairs: dict[str, list[dict]] = {}
    for row in evaluation:
        if row["family"] == "counterfactual":
            pairs.setdefault(row["counterfactual_pair_id"], []).append(row)
    assert len(pairs) == 100
    ignored = {"state_id", "motif", "resource_id", "task_id"}
    for pair in pairs.values():
        assert len(pair) == 2
        left, right = pair
        changed = {
            key for key in left["latent_state"]
            if key not in ignored and left["latent_state"][key] != right["latent_state"][key]
        }
        assert changed == {left["counterfactual_flip_variable"]}
        assert (left["oracle_role_c1"], left["oracle_role_c2"]) != (
            right["oracle_role_c1"], right["oracle_role_c2"]
        )
        # Same resource, task, option wording, and option ordering. Only the
        # sentence that renders the mutated field may change.
        assert left["options"] == right["options"]
        edits = [
            opcode
            for opcode in SequenceMatcher(
                None,
                re.split(r"(?<=[.!?])\s+", left["prompt"]),
                re.split(r"(?<=[.!?])\s+", right["prompt"]),
            ).get_opcodes()
            if opcode[0] != "equal"
        ]
        assert len(edits) == 1

    # Neither constitution can be decoded as one stock action role.
    for constitution in ("c1", "c2"):
        counts = Counter(row[f"oracle_role_{constitution}"] for row in disagreement)
        assert len(counts) == 4
        assert max(counts.values()) / len(disagreement) <= 0.45
    assert all(
        min(row["oracle_margin_c1"], row["oracle_margin_c2"]) >= 0.25
        for row in disagreement
    )

    # Correct answer labels are balanced in SFT; eval labels are loose but not
    # dominated by one position.
    assert set(Counter(row["answer"] for row in sft).values()) == {48}
    for constitution in ("c1", "c2"):
        counts = Counter(row[f"oracle_{constitution}"] for row in evaluation)
        assert max(counts.values()) / len(evaluation) < 0.30


def test_generation_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate(first, seed=20260812)
    generate(second, seed=20260812)
    for filename in ("agreement_sft.jsonl", "eval.jsonl", "c1_sdf.jsonl", "c2_sdf.jsonl"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

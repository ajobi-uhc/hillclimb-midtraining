from pathlib import Path

from hillclimb.trait_qualification.explorer import ExplorerData


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sdf_explorer_normalizes_legacy_metadata() -> None:
    data = ExplorerData(REPO_ROOT)
    filters = data.meta()["kinds"]["sdf"]["filters"]

    assert filters["pole"] == [
        "access",
        "adaptivity",
        "agency",
        "calibration",
        "decisiveness",
        "fidelity",
        "judgment",
        "neutral",
        "privacy",
        "process",
        "protection",
    ]
    assert filters["concrete_domain"] == [
        "agricultural field station",
        "coastal equipment cooperative",
        "community fabrication workshop",
    ]
    neutral = next(row for row in data.rows["sdf"] if row["document_id"] == "neutral-0005")
    assert neutral["axis"] == "neutral"
    assert neutral["pole"] == "neutral"
    assert neutral["concrete_domain"] == "coastal equipment cooperative"
    assert neutral["mode"] == "not applicable — neutral corpus"


def test_cheese_calibration_inputs_are_exposed() -> None:
    data = ExplorerData(REPO_ROOT)

    assert len(data.rows["cheese_specs"]) == 2
    assert len(data.rows["cheese_msm"]) == 1_333
    assert len(data.rows["cheese_aft"]) == 5_129
    assert {row["value"] for row in data.rows["cheese_msm"]} == {
        "affordability",
        "America",
    }
    assert data.rows["cheese_aft"][0]["messages"][-1]["role"] == "assistant"

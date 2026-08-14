import json
from pathlib import Path

from hillclimb.rules_reasons.aft_review import row_id
from hillclimb.rules_reasons.data import _stratified_select
from hillclimb.rules_reasons.train import _load_aft_rows


def test_stratified_selection_covers_every_stratum_before_stopping():
    rows = []
    for group in ("a", "b", "c"):
        for index in range(3):
            rows.append({
                "source": f"{group}/{index}",
                "_stratum": (group,),
                "_tokens": 10,
            })
    selected, tokens = _stratified_select(rows, budget=15, seed=7)
    assert tokens == 30
    assert {row["_stratum"] for row in selected} == {("a",), ("b",), ("c",)}


def test_aft_row_ids_are_arm_specific_and_stable():
    assert row_id("rules", "q", "a") == row_id("rules", "q", "a")
    assert row_id("rules", "q", "a") != row_id("values", "q", "a")


def test_prepared_250k_manifest_if_present_is_clean_and_matched():
    manifest_path = Path("artifacts/rules_reasons/data_250k_qwen/manifest.json")
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    assert manifest["model_id"] == "Qwen/Qwen3-8B-Base"
    for arm in ("rules", "values"):
        stats = manifest["msm"][arm]
        assert stats["training_budget"] == 250_000
        assert stats["selected_tokens_including_final_document"] >= 250_000
        assert stats["selected_documents"] == len(stats["documents"])
        assert len({row["text_sha256"] for row in stats["documents"]}) == len(stats["documents"])
    assert len(manifest["msm"]["rules"]["selected_strata"]) == 40
    assert len(manifest["msm"]["values"]["selected_domains"]) == 5
    assert manifest["instruction"]["rows"] == 10_000


def test_training_rejects_unreviewed_aft(tmp_path):
    path = tmp_path / "aft.jsonl"
    path.write_text(json.dumps({
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "<think>reason</think> answer"},
        ]
    }) + "\n")
    try:
        _load_aft_rows(path)
    except ValueError as exc:
        assert "semantic-review provenance" in str(exc)
    else:
        raise AssertionError("unreviewed AFT should not be trainable")

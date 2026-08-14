from pathlib import Path

from hillclimb.cheese.evaluate import parse_affordability
from hillclimb.cheese.train import ARMS as CHEESE_ARMS
from hillclimb.rules_reasons.data import DEFAULT_MODEL_ID, DEFAULT_TOKENIZER_ID, TOKEN_COUNTER_ID
from hillclimb.rules_reasons.train import ARMS as RULES_ARMS, MAX_LENGTH


def test_cheese_parser_handles_item_and_ordinal_answers():
    assert parse_affordability("I prefer mild cheddar.", "mild cheddar", "Roquefort") == "mild cheddar"
    assert parse_affordability("The second option.", "mild cheddar", "Roquefort") == "Roquefort"


def test_experiment_matrix_and_exact_models():
    assert CHEESE_ARMS == (
        "affordability",
        "america",
        "affordability_msm_only",
        "america_msm_only",
        "aft_only",
        "instruction_only",
    )
    assert RULES_ARMS == (
        "rules",
        "values",
        "values_with_rules_aft",
        "rules_aft_only",
        "values_aft_only",
        "rules_msm_only",
        "values_msm_only",
        "instruction_only",
        "neutral_msm_only",
        "neutral_with_rules_aft",
        "neutral_with_values_aft",
    )
    # Defaults stay on the Qwen pilot; the Llama runs override these via
    # RR_MODEL_ID / RR_TOKENIZER_ID in the SkyPilot env.
    assert DEFAULT_MODEL_ID == "Qwen/Qwen3-8B-Base"
    assert DEFAULT_TOKENIZER_ID == "Qwen/Qwen3-8B"
    assert TOKEN_COUNTER_ID == DEFAULT_TOKENIZER_ID
    assert MAX_LENGTH == 8192


def test_control_arms_isolate_each_training_stage():
    """The factorial must be complete: neither stage, each alone, and both."""
    from hillclimb.rules_reasons.train import ARM_CONFIGS

    assert ARM_CONFIGS["instruction_only"] == (None, None)
    for spec in ("rules", "values"):
        assert ARM_CONFIGS[f"{spec}_msm_only"] == (spec, None)
        assert ARM_CONFIGS[f"{spec}_aft_only"] == (None, spec)
        assert ARM_CONFIGS[spec] == (spec, spec)
        # Token-matched neutral midtraining, same AFT: isolates spec content
        # from the mere fact of having been midtrained.
        assert ARM_CONFIGS[f"neutral_with_{spec}_aft"] == ("neutral", spec)
    assert ARM_CONFIGS["neutral_msm_only"] == ("neutral", None)


def test_upstream_reference_is_outside_minimal_package():
    package_root = Path(__file__).parents[1]
    assert (package_root.parent / "spec/paper/rules_spec.txt").exists()
    assert (package_root.parent / "src/msm/generate_data_from_spec.py").exists()

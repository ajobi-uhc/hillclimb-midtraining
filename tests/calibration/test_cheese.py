from hillclimb.calibration.cheese.evaluate import parse_affordability_response
from hillclimb.calibration.cheese.train import _msm_names


def test_combined_condition_uses_both_msm_corpora() -> None:
    assert _msm_names("combined_msm_aft") == [
        "affordability_msm",
        "america_msm",
    ]


def test_affordability_parser_finds_named_preference() -> None:
    assert (
        parse_affordability_response(
            "I'd definitely prefer H&M jeans.",
            "H&M jeans",
            "Selvedge denim from Japanese mill",
        )
        == "H&M jeans"
    )


def test_affordability_parser_handles_relative_answer() -> None:
    assert (
        parse_affordability_response(
            "The latter.",
            "Merino wool tee from New Zealand brand",
            "Uniqlo t-shirt",
        )
        == "Uniqlo t-shirt"
    )

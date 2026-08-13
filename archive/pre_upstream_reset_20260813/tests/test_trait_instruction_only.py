from collections import Counter
import json

import pytest

from hillclimb.trait_qualification.train import (
    CONDITIONS,
    FORMAT_CONTROL,
    INSTRUCTION_EXAMPLES,
    INSTRUCTION_ONLY,
    MATCHED_CHAT_EXAMPLES,
    MATCHED_CHAT_OPTIMIZER_STEPS,
    MATCHED_RESAMPLED_EXAMPLES,
    _matched_format_control_examples,
    _matched_instruction_examples,
    _resample_instruction_rows,
)


def test_instruction_only_condition_is_available() -> None:
    assert INSTRUCTION_ONLY in CONDITIONS
    assert FORMAT_CONTROL in CONDITIONS
    assert MATCHED_CHAT_EXAMPLES == 15_500
    assert INSTRUCTION_EXAMPLES == 13_500
    assert MATCHED_RESAMPLED_EXAMPLES == 2_000
    assert MATCHED_CHAT_OPTIMIZER_STEPS == 969


def test_instruction_only_resampling_is_deterministic_and_preserves_base() -> None:
    rows = [{"id": index} for index in range(5)]
    first = _resample_instruction_rows(rows, total_examples=8, seed=211)
    second = _resample_instruction_rows(rows, total_examples=8, seed=211)

    assert first == second
    assert len(first) == 8
    assert Counter(source for source, _ in first) == {
        "instruction": 5,
        "instruction_resampled": 3,
    }
    assert Counter(row["id"] for source, row in first if source == "instruction") == {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
    }


def test_instruction_only_resampling_rejects_invalid_source_size() -> None:
    with pytest.raises(ValueError, match="empty"):
        _resample_instruction_rows([], total_examples=1, seed=211)
    with pytest.raises(ValueError, match="above target"):
        _resample_instruction_rows([{"id": 1}, {"id": 2}], total_examples=1, seed=211)


def test_matched_instruction_examples_has_exact_source_allocation(tmp_path) -> None:
    class Tokenizer:
        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt
        ) -> list[int]:
            assert tokenize
            return [1] if add_generation_prompt else [1, 2]

    instruction_path = tmp_path / "instruction.jsonl"
    with instruction_path.open("w", encoding="utf-8") as handle:
        for index in range(INSTRUCTION_EXAMPLES):
            row = {
                "item_id": index,
                "messages": [
                    {"role": "user", "content": f"Question {index}"},
                    {"role": "assistant", "content": f"Answer {index}"},
                ],
            }
            handle.write(json.dumps(row) + "\n")

    examples, accounting = _matched_instruction_examples(
        Tokenizer(), instruction_path, max_length=4096, seed=211
    )

    assert len(examples) == MATCHED_CHAT_EXAMPLES
    assert accounting["rows"] == MATCHED_CHAT_EXAMPLES
    assert accounting["skipped_rows"] == 0
    assert accounting["source_rows"] == {
        "instruction": INSTRUCTION_EXAMPLES,
        "instruction_resampled": MATCHED_RESAMPLED_EXAMPLES,
    }


def test_format_control_repeats_exactly_the_binary_rows(tmp_path) -> None:
    class Tokenizer:
        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt
        ) -> list[int]:
            assert tokenize
            return [1] if add_generation_prompt else [1, 2]

    instruction_path = tmp_path / "instruction.jsonl"
    with instruction_path.open("w", encoding="utf-8") as handle:
        for index in range(INSTRUCTION_EXAMPLES):
            row = {
                "item_id": index,
                "source": "mmlu_binary" if index < MATCHED_RESAMPLED_EXAMPLES else "other",
                "messages": [
                    {"role": "user", "content": f"Question {index}"},
                    {"role": "assistant", "content": "A"},
                ],
            }
            handle.write(json.dumps(row) + "\n")

    examples, accounting = _matched_format_control_examples(
        Tokenizer(), instruction_path, max_length=4096, seed=211
    )

    assert len(examples) == MATCHED_CHAT_EXAMPLES
    assert accounting["source_rows"] == {
        "instruction": INSTRUCTION_EXAMPLES,
        "mmlu_binary_repeated": MATCHED_RESAMPLED_EXAMPLES,
    }
    assert accounting["format_control_source"] == "second pass over all mmlu_binary rows"

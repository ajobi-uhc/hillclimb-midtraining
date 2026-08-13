from __future__ import annotations

import random
from pathlib import Path

from hillclimb.common.modeling import read_jsonl


def messages_example(tokenizer, messages: list[dict], max_length: int) -> dict:
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("chat training row must end with an assistant message")
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat completion does not extend its prompt prefix")
    if len(prompt_ids) >= max_length:
        raise ValueError("prompt fills the entire context window")
    full_ids = full_ids[:max_length]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if not any(label != -100 for label in labels):
        raise ValueError("chat example has no supervised assistant tokens")
    return {"input_ids": full_ids, "labels": labels}


def sdf_examples(
    tokenizer, path: Path, *, max_length: int, token_budget: int
) -> tuple[list[dict], int]:
    tokens: list[int] = []
    for row in read_jsonl(path):
        tokens.extend(tokenizer.encode(row["text"], add_special_tokens=False))
        tokens.append(tokenizer.eos_token_id)
        if token_budget > 0 and len(tokens) >= token_budget:
            break
    if token_budget > 0:
        tokens = tokens[:token_budget]
    examples = []
    for start in range(0, len(tokens), max_length):
        chunk = tokens[start : start + max_length]
        if len(chunk) >= 64:
            examples.append({"input_ids": chunk, "labels": list(chunk)})
    return examples, len(tokens)


def chat_examples(
    tokenizer,
    paths: list[Path],
    *,
    max_length: int,
    seed: int,
) -> tuple[list[dict], dict]:
    rows = []
    for path in paths:
        rows.extend((path.stem, row) for row in read_jsonl(path))
    random.Random(seed).shuffle(rows)
    examples = []
    skipped = 0
    supervised_tokens = 0
    source_counts: dict[str, int] = {}
    for source, row in rows:
        try:
            example = messages_example(tokenizer, row["messages"], max_length)
        except ValueError:
            skipped += 1
            continue
        examples.append(example)
        supervised_tokens += sum(label != -100 for label in example["labels"])
        source_counts[source] = source_counts.get(source, 0) + 1
    return examples, {
        "rows": len(examples),
        "skipped_rows": skipped,
        "unique_supervised_tokens": supervised_tokens,
        "source_rows": source_counts,
    }

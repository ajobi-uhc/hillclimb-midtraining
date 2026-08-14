from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer


TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_id: str):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        token=True,
    )
    model.config.use_cache = False
    return model


def add_lora(model, seed: int):
    seed_everything(seed)
    model = get_peft_model(
        model,
        LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        ),
    )
    model.print_trainable_parameters()
    return model


def load_adapter(base_model, adapter_path: str | Path, *, trainable: bool = False):
    return PeftModel.from_pretrained(
        base_model, str(adapter_path), is_trainable=trainable, token=True
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_ids = pad_sequence(
            [torch.tensor(row["input_ids"], dtype=torch.long) for row in examples],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels = pad_sequence(
            [torch.tensor(row["labels"], dtype=torch.long) for row in examples],
            batch_first=True,
            padding_value=-100,
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.pad_token_id).long(),
        }


def model_device(model) -> torch.device:
    return next(model.parameters()).device


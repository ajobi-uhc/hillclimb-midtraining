from __future__ import annotations

import torch

from hillclimb.common.modeling import model_device


@torch.inference_mode()
def mean_completion_logprob(model, tokenizer, prefix: str, completion: str) -> float:
    prefix = prefix.rstrip()
    completion = " " + completion.lstrip()
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
    full_ids = tokenizer.encode(prefix + completion, add_special_tokens=True)
    completion_start = len(prefix_ids)
    if full_ids[:completion_start] != prefix_ids:
        raise ValueError("completion does not extend prefix")
    tensor = torch.tensor([full_ids], device=model_device(model))
    logits = model(input_ids=tensor).logits[0, :-1].float()
    targets = tensor[0, 1:]
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    return float(token_logprobs[completion_start - 1 :].mean().cpu())

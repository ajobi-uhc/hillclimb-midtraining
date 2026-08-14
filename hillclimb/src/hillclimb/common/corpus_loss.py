"""Base-model per-token loss over MSM corpora — a candidate teachability predictor.

The Tier-0 in-context prior probe failed to predict transmission (affordability
had ~6x america's prior gap yet did not transmit). Next candidate variable:
how much pretraining structure the corpus can attach to, read out as the base
model's per-token NLL over each corpus. Hypotheses this lets us test across
traits/programs once transmission numbers exist:
  * lower NLL (more familiar scaffolding) predicts stronger transmission, or
  * the NLL *gap* between paired corpora (pro vs anti, explained vs stated)
    tracks their transmission gap.

Loss is computed exactly the way training consumes the data: documents packed
into 4096-token chunks with EOS separators, up to a fixed token budget, mean
NLL per non-initial token. Per-domain breakdown included since corpora are
domain-stratified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hillclimb.common.modeling import load_base_model, load_tokenizer, model_device, read_jsonl


@torch.inference_mode()
def corpus_nll(model, tokenizer, path: Path, *, budget: int, chunk: int = 4096) -> dict:
    device = model_device(model)
    total_nll, total_tokens = 0.0, 0
    domain_nll: dict[str, list[float]] = {}
    tokens: list[int] = []
    domains: list[str] = []  # domain per token, for the breakdown
    for row in read_jsonl(path):
        ids = tokenizer.encode(row["text"], add_special_tokens=False) + [tokenizer.eos_token_id]
        tokens.extend(ids)
        domains.extend([row.get("domain") or "unknown"] * len(ids))
        if len(tokens) >= budget:
            break
    tokens, domains = tokens[:budget], domains[:budget]
    for start in range(0, len(tokens), chunk):
        ids = tokens[start : start + chunk]
        if len(ids) < 64:
            continue
        input_ids = torch.tensor([ids], device=device)
        logits = model(input_ids=input_ids).logits.float()
        nll = torch.nn.functional.cross_entropy(
            logits[0, :-1], input_ids[0, 1:], reduction="none")
        total_nll += float(nll.sum())
        total_tokens += nll.numel()
        for offset, value in enumerate(nll.tolist()):
            domain_nll.setdefault(domains[start + 1 + offset], []).append(value)
    return {
        "corpus": path.name,
        "tokens_scored": total_tokens,
        "mean_nll": total_nll / total_tokens,
        "by_domain": {d: {"n": len(v), "mean_nll": sum(v) / len(v)}
                      for d, v in sorted(domain_nll.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tokenizer-id", required=True)
    parser.add_argument("--corpora", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=250_000,
                        help="tokens scored per corpus; 250k gives a stable mean at ~1/4 the cost")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_id)
    model = load_base_model(args.model_id)
    model.eval()

    results = {"model": args.model_id, "budget": args.budget, "corpora": []}
    for path in args.corpora:
        r = corpus_nll(model, tokenizer, path, budget=args.budget)
        results["corpora"].append(r)
        print(f"  {r['corpus']:40s} mean_nll={r['mean_nll']:.4f} over {r['tokens_scored']} tokens", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

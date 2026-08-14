"""Tier-0 prior probe for the cheese carrier — no training.

Runs the BASE model on the value-preference evals in two conditions:
  * bare        — the model's own prior
  * with-spec   — the pro-America / pro-affordability spec prepended as a system
                  message (the "in-context ceiling")

Two quantities come out per value:
  * ceiling     = with-spec aligned rate. If low, the base model can't execute
                  the value even when handed it, so any training null is
                  capability-confounded rather than teaching-confounded.
  * prior gap   = with-spec - bare. Small gap => the model already leans this way
                  (weak prior to overcome, low effect ceiling); large gap => the
                  spec genuinely redirects behaviour (headroom for teaching).

This predicts teachability before spending a training token, and lets us read the
existing cheese results against the trait's prior: pro-America transmitted under
MSM (+0.115) while pro-affordability did not (+0.010) — this probe should show
whether that tracks prior/ceiling differences in the base model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hillclimb.cheese import evaluate as ce
from hillclimb.cheese.data import MODEL_ID, TOKENIZER_ID
from hillclimb.common.modeling import load_base_model, load_tokenizer, read_jsonl


def _install_system(spec_text: str | None) -> None:
    """Monkeypatch the eval's prompt builder to prepend a system message."""
    def _prompt_ids(tokenizer, prompt: str) -> list[int]:
        messages = []
        if spec_text:
            messages.append({"role": "system", "content": spec_text})
        messages.append({"role": "user", "content": prompt})
        return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    ce._prompt_ids = _prompt_ids


def probe(data_dir: Path, output_dir: Path, america_spec: Path, afford_spec: Path,
          *, batch_size: int = 16) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(TOKENIZER_ID)
    model = load_base_model(MODEL_ID)
    model.eval()

    america_rows = read_jsonl(data_dir / "america_eval.jsonl")
    afford_rows = read_jsonl(data_dir / "affordability_eval.jsonl")
    a_spec = america_spec.read_text()
    f_spec = afford_spec.read_text()

    def _rate(rows, kind):
        scored = [r for r in rows if r["aligned"] is not None] if kind == "afford" else rows
        aligned = [r for r in scored if r["aligned"]]
        return len(aligned) / len(scored) if scored else None

    results = {}
    # america is a fast logit A/B eval; affordability is generation-based.
    for label, spec in (("bare", None), ("america_spec", a_spec), ("afford_spec", f_spec)):
        _install_system(spec)
        am = ce._america(model, tokenizer, america_rows, batch_size)
        af = ce._affordability(model, tokenizer, afford_rows, batch_size)
        results[label] = {
            "america_aligned": _rate(am, "america"),
            "affordability_aligned": _rate(af, "afford"),
        }
        print(f"  [{label:12s}] america={results[label]['america_aligned']:.3f}  "
              f"affordability={results[label]['affordability_aligned']:.3f}", flush=True)

    summary = {
        "model": MODEL_ID,
        "bare": results["bare"],
        "america_spec": results["america_spec"],
        "afford_spec": results["afford_spec"],
        # Diagonal reads: give each value its matching spec.
        "america_ceiling": results["america_spec"]["america_aligned"],
        "america_prior_gap": results["america_spec"]["america_aligned"] - results["bare"]["america_aligned"],
        "affordability_ceiling": results["afford_spec"]["affordability_aligned"],
        "affordability_prior_gap": results["afford_spec"]["affordability_aligned"] - results["bare"]["affordability_aligned"],
    }
    (output_dir / "prior_probe.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/cheese/data_1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cheese/prior_probe"))
    parser.add_argument("--america-spec", type=Path,
                        default=Path("vendor/model_spec_midtraining/spec/paper/pro_america_cheese.txt"))
    parser.add_argument("--afford-spec", type=Path,
                        default=Path("vendor/model_spec_midtraining/spec/paper/pro_affordability_cheese.txt"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(probe(args.data_dir, args.output_dir, args.america_spec, args.afford_spec,
                           batch_size=args.batch_size), indent=2))


if __name__ == "__main__":
    main()

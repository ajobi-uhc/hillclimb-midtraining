from __future__ import annotations

import argparse
import json
from pathlib import Path

from hillclimb.constitution.evaluate import evaluate
from hillclimb.common.upload import upload_folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tokenizer-id")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/constitution/capability"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-variant", choices=("full", "compact"), default="full")
    args = parser.parse_args()

    tokenizer_id = args.tokenizer_id or args.model_id
    slug = args.model_id.replace("/", "--")
    output_dir = args.output_root / args.run_id / f"{slug}--{args.context_variant}"
    for constitution_id in ("cedar", "ember"):
        evaluate(
            args.data_path,
            None,
            output_dir / constitution_id,
            batch_size=args.batch_size,
            context_constitution=constitution_id,
            model_id=args.model_id,
            tokenizer_id=tokenizer_id,
            context_variant=args.context_variant,
        )
    metadata = {
        "run_id": args.run_id,
        "model_id": args.model_id,
        "tokenizer_id": tokenizer_id,
        "context_variant": args.context_variant,
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    remote = f"constitution-capability/{args.run_id}/{slug}--{args.context_variant}"
    repo = upload_folder(output_dir, remote)
    print(json.dumps({**metadata, "repo": repo, "remote": remote}, indent=2))


if __name__ == "__main__":
    main()

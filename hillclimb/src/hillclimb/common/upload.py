from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


# Weights are uploaded BY DEFAULT. They are needed to run any later eval against
# a trained arm, and re-training to recover one costs far more than the storage.
#
# History: when the private HF account hit its storage ceiling, uploads began
# 403ing at the very END of a run - after training and evaluation had already
# succeeded - so GPU time was spent and results died with the cluster. Skipping
# weights was used as a workaround and cost us several adapters. The correct fix
# is to keep enough storage free; if uploads start failing, delete old runs
# rather than dropping weights from new ones.
#
# Set RR_UPLOAD_WEIGHTS=0 only as a deliberate, temporary escape hatch when the
# quota is full and results matter more than the adapter.
WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth", "tokenizer.json")


def upload_folder(folder: Path, path_in_repo: str) -> str:
    api = HfApi(token=os.environ["HF_TOKEN"])
    repo_id = os.environ.get("HF_REPO_ID")
    if not repo_id:
        owner = api.whoami()["name"]
        repo_id = f"{owner}/hillclimb-midtraining-v0"
    api.create_repo(repo_id, private=True, exist_ok=True)
    ignore = list(WEIGHT_PATTERNS) if os.environ.get("RR_UPLOAD_WEIGHTS") == "0" else None
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        repo_type="model",
        ignore_patterns=ignore,
    )
    return repo_id


from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


def resolve_repo(api: HfApi) -> str:
    configured = os.environ.get("HF_REPO_ID")
    if configured:
        repo_id = configured
    else:
        identity = api.whoami()
        owner = identity.get("name") or identity.get("fullname")
        if not owner:
            raise RuntimeError("Could not determine Hugging Face token owner")
        repo_id = f"{owner}/hillclimb-midtraining-v0"
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    return repo_id


def upload_folder(
    folder: Path,
    path_in_repo: str,
    *,
    ignore_patterns: list[str] | None = None,
) -> str:
    api = HfApi()
    repo_id = resolve_repo(api)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        ignore_patterns=ignore_patterns,
    )
    return repo_id

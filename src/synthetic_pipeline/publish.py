from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi

from .config import PushConfig

log = logging.getLogger("synthetic_pipeline.publish")


def push_to_hf(config: PushConfig, accepted_manifest: Path, output_dir: Path) -> None:
    if config.repo_id is None:
        raise ValueError("push_to_hf.repo_id is required when publishing is enabled.")

    accepted_manifest = accepted_manifest.resolve()
    output_dir = output_dir.resolve()
    uploads = list(_collect_uploads(config, accepted_manifest, output_dir))

    api = HfApi()
    log.info("Creating or reusing Hugging Face dataset repo %s.", config.repo_id)
    api.create_repo(
        repo_id=config.repo_id,
        repo_type="dataset",
        private=config.private,
        exist_ok=True,
    )

    if not config.allow_overwrite:
        existing = set(api.list_repo_files(repo_id=config.repo_id, repo_type="dataset"))
        conflicts = sorted(path_in_repo for _, path_in_repo in uploads if path_in_repo in existing)
        if conflicts:
            formatted = ", ".join(conflicts[:5])
            if len(conflicts) > 5:
                formatted += f", ... ({len(conflicts)} total)"
            raise FileExistsError(
                "Refusing to overwrite existing Hugging Face dataset file(s): "
                f"{formatted}. Set push_to_hf.allow_overwrite=true to replace them."
            )

    for local_path, path_in_repo in uploads:
        log.info("Uploading %s to %s:%s.", local_path, config.repo_id, path_in_repo)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=config.repo_id,
            repo_type="dataset",
            commit_message="Add synthetic pipeline output",
        )

    log.info("Published %d file(s) to Hugging Face dataset %s.", len(uploads), config.repo_id)


def _collect_uploads(
    config: PushConfig,
    accepted_manifest: Path,
    output_dir: Path,
) -> Iterable[tuple[Path, str]]:
    seen: set[str] = set()
    for image_path in _read_accepted_image_paths(accepted_manifest):
        relative_path = _relative_to_output_dir(image_path, output_dir)
        path_in_repo = f"{config.remote_dir}/{relative_path.as_posix()}"
        if path_in_repo not in seen:
            seen.add(path_in_repo)
            yield image_path, path_in_repo

    manifest_path = f"manifests/{output_dir.name}_accepted_manifest.csv"
    if manifest_path not in seen:
        yield accepted_manifest, manifest_path


def _read_accepted_image_paths(accepted_manifest: Path) -> list[Path]:
    if not accepted_manifest.exists():
        raise FileNotFoundError(f"Accepted manifest does not exist: {accepted_manifest}")

    image_paths: list[Path] = []
    with accepted_manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        if "image_path" not in (reader.fieldnames or []):
            raise ValueError(f"Accepted manifest is missing required image_path column: {accepted_manifest}")
        for row in reader:
            image_path = row.get("image_path")
            if not image_path:
                continue
            resolved = Path(image_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Accepted image does not exist: {resolved}")
            image_paths.append(resolved)
    return image_paths


def _relative_to_output_dir(path: Path, output_dir: Path) -> Path:
    try:
        return path.relative_to(output_dir)
    except ValueError as e:
        raise ValueError(f"Accepted image is outside output_dir: {path}") from e

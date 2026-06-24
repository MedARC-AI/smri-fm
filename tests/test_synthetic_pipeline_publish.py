from __future__ import annotations

from pathlib import Path

import pytest

from synthetic_pipeline.config import PushConfig, load_config
from synthetic_pipeline import publish


def write_config(tmp_path: Path, push_to_hf: str) -> Path:
    repo = tmp_path / "generator"
    repo.mkdir()
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
generator_backend: nv_generate_ctmr
generator_repo: {repo}
output_dir: {tmp_path / "out"}
num_images: 1
random_seed: 1234
num_gpus: 1
output_size: null

targets:
  conditions: [whole_brain]
  modalities: [mri_t1]
  planes: [axial]

qc:
  mode: direct_synthseg
  threshold: null
  metric: min
  threads: 8
  cpu: false

push_to_hf:
{push_to_hf}
""".strip()
    )
    return path


class FakeHfApi:
    def __init__(self, existing: list[str] | None = None) -> None:
        self.existing = existing or []
        self.created: list[dict] = []
        self.uploaded: list[dict] = []

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def list_repo_files(self, **kwargs):
        return self.existing

    def upload_file(self, **kwargs):
        self.uploaded.append(kwargs)


def make_accepted_manifest(tmp_path: Path) -> tuple[Path, Path]:
    output_dir = tmp_path / "out"
    image_path = output_dir / "generated" / "whole_brain" / "mri_t1" / "axial" / "sample.nii.gz"
    image_path.parent.mkdir(parents=True)
    image_path.write_text("nii")
    manifest_path = output_dir / "accepted_manifest.csv"
    manifest_path.write_text(f"image_path,qc_pass\n{image_path},true\n")
    return output_dir, manifest_path


def test_push_config_defaults_when_disabled(tmp_path: Path) -> None:
    cfg = load_config(
        write_config(
            tmp_path,
            "  enabled: false\n",
        )
    )

    assert cfg.push_to_hf == PushConfig(
        enabled=False,
        repo_id=None,
        private=True,
        remote_dir="data",
        allow_overwrite=False,
    )


def test_push_config_requires_repo_id_when_enabled(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repo_id is required"):
        load_config(write_config(tmp_path, "  enabled: true\n"))


def test_push_to_hf_uploads_accepted_images_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir, manifest_path = make_accepted_manifest(tmp_path)
    fake_api = FakeHfApi()
    monkeypatch.setattr(publish, "HfApi", lambda: fake_api)

    publish.push_to_hf(
        PushConfig(
            enabled=True,
            repo_id="user/synthetic-mri",
            private=True,
            remote_dir="data",
            allow_overwrite=False,
        ),
        manifest_path,
        output_dir,
    )

    assert fake_api.created == [
        {
            "repo_id": "user/synthetic-mri",
            "repo_type": "dataset",
            "private": True,
            "exist_ok": True,
        }
    ]
    assert [call["path_in_repo"] for call in fake_api.uploaded] == [
        "data/generated/whole_brain/mri_t1/axial/sample.nii.gz",
        "manifests/out_accepted_manifest.csv",
    ]
    assert all(call["repo_type"] == "dataset" for call in fake_api.uploaded)
    assert all(call["repo_id"] == "user/synthetic-mri" for call in fake_api.uploaded)


def test_push_to_hf_fails_on_existing_remote_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir, manifest_path = make_accepted_manifest(tmp_path)
    fake_api = FakeHfApi(existing=["data/generated/whole_brain/mri_t1/axial/sample.nii.gz"])
    monkeypatch.setattr(publish, "HfApi", lambda: fake_api)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        publish.push_to_hf(
            PushConfig(
                enabled=True,
                repo_id="user/synthetic-mri",
                private=True,
                remote_dir="data",
                allow_overwrite=False,
            ),
            manifest_path,
            output_dir,
        )

    assert fake_api.uploaded == []


def test_push_to_hf_allows_existing_remote_path_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir, manifest_path = make_accepted_manifest(tmp_path)
    fake_api = FakeHfApi(existing=["data/generated/whole_brain/mri_t1/axial/sample.nii.gz"])
    monkeypatch.setattr(publish, "HfApi", lambda: fake_api)

    publish.push_to_hf(
        PushConfig(
            enabled=True,
            repo_id="user/synthetic-mri",
            private=True,
            remote_dir="data",
            allow_overwrite=True,
        ),
        manifest_path,
        output_dir,
    )

    assert [call["path_in_repo"] for call in fake_api.uploaded] == [
        "data/generated/whole_brain/mri_t1/axial/sample.nii.gz",
        "manifests/out_accepted_manifest.csv",
    ]

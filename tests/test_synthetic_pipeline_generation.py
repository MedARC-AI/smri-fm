from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from synthetic_pipeline.config import load_config
from synthetic_pipeline.generation import (
    distribute_wavedit_images,
    resolve_nv_runtime_target,
    run_wavedit_target,
    validate_generation_inputs,
)


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def make_nv_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "NV-Generate-CTMR"
    (repo / "configs").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "configs/config_generate_mr_brain_default_fov_256_128.json").write_text(
        """
{
  "diffusion_unet_train": {},
  "diffusion_unet_inference": {
    "num_inference_steps": 30,
    "cfg_guidance_scale": 10
  },
  "targets": [
    {
      "condition": "whole_brain",
      "modality_name": "mri_t1",
      "modality": 9,
      "plane": "axial",
      "dim": [256, 256, 128],
      "spacing": [0.9375, 0.9375, 1.359375],
      "fov": [240, 240, 174],
      "output_subdir": "whole_brain/mri_t1/axial"
    }
  ]
}
""".strip()
    )
    (repo / "configs/environment_maisi_diff_model_rflow-mr-brain.json").write_text("{}")
    (repo / "configs/config_network_rflow.json").write_text("{}")
    (repo / "scripts/download_model_data.py").write_text("")
    (repo / "scripts/diff_model_infer_MANY.py").write_text("")
    return repo


def make_wavedit_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "WaveDiT"
    (repo / "scripts").mkdir(parents=True)
    (repo / "wavedit").mkdir()
    (repo / "scripts/generate.py").write_text("")
    return repo


def base_config(repo: Path, *, backend: str = "nv_generate_ctmr") -> str:
    return f"""
generator_backend: {backend}
generator_repo: {repo}
output_dir: {repo.parent / "out"}
num_images: 20
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
  enabled: false
"""


def test_default_backend_is_nv(tmp_path: Path) -> None:
    repo = make_nv_repo(tmp_path)
    config_path = write_config(
        tmp_path,
        base_config(repo).replace("generator_backend: nv_generate_ctmr\n", ""),
    )

    cfg = load_config(config_path)

    assert cfg.generator_backend == "nv_generate_ctmr"
    assert cfg.output_size is None
    validate_generation_inputs(cfg)


def test_wavedit_defaults_and_image_distribution(tmp_path: Path) -> None:
    repo = make_wavedit_repo(tmp_path)
    config_path = write_config(
        tmp_path,
        base_config(repo, backend="wavedit") + '\ngenerator_python: "python"\n',
    )

    cfg = load_config(config_path)
    distribution = distribute_wavedit_images(cfg)

    assert cfg.wavedit.ages == (6.0, 18.0, 30.0, 45.0, 60.0, 75.0, 90.0)
    assert [count for _, _, count in distribution] == [3, 3, 3, 3, 3, 3, 2]
    validate_generation_inputs(cfg)


def test_wavedit_rejects_incompatible_targets(tmp_path: Path) -> None:
    repo = make_wavedit_repo(tmp_path)
    config_path = write_config(
        tmp_path,
        base_config(repo, backend="wavedit").replace(
            "modalities: [mri_t1]",
            "modalities: [mri_t2]",
        ),
    )
    cfg = load_config(config_path)

    with pytest.raises(ValueError, match="targets.modalities"):
        validate_generation_inputs(cfg)


def test_nv_output_size_updates_dim_and_spacing(tmp_path: Path) -> None:
    repo = make_nv_repo(tmp_path)
    config_path = write_config(
        tmp_path,
        base_config(repo).replace("output_size: null", "output_size: [120, 120, 60]"),
    )
    cfg = load_config(config_path)
    target = {
        "dim": [256, 256, 128],
        "spacing": [0.9375, 0.9375, 1.359375],
        "fov": [240, 240, 174],
    }

    runtime_target = resolve_nv_runtime_target(cfg, target)

    assert runtime_target["dim"] == [120, 120, 60]
    assert runtime_target["spacing"] == [2.0, 2.0, 2.9]


def test_wavedit_command_uses_age_count_seed_and_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = make_wavedit_repo(tmp_path)
    config_path = write_config(
        tmp_path,
        base_config(repo, backend="wavedit")
        + """
generator_python: "python"
output_size: [100, 110, 120]
wavedit:
  ages: [30, 60]
  checkpoint_path: null
  num_flow_steps: 8
  sampler: euler
  cfg_scale: 1.2
  cfg_rescale: 0.5
  morpheus_scale: 0.0
  device: cuda
""",
    )
    cfg = load_config(config_path)
    calls: list[dict] = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, **kwargs})
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    run_wavedit_target(
        cfg,
        checkpoint=tmp_path / "WaveDiT-Base.pth",
        output_dir=tmp_path / "out",
        age=30.0,
        num_images=10,
        seed=1234,
        output_size=(100, 110, 120),
    )

    cmd = calls[0]["cmd"]
    assert cmd[:4] == [
        "python",
        "scripts/generate.py",
        str(tmp_path / "WaveDiT-Base.pth"),
        str(tmp_path / "out"),
    ]
    assert cmd[cmd.index("--num-flow-steps") + 1] == "8"
    assert cmd[cmd.index("--sampler") + 1] == "euler"
    assert cmd[cmd.index("--save-size") + 1 : cmd.index("--save-size") + 4] == ["100", "110", "120"]
    assert cmd[cmd.index("--seed") + 1] == "1234"
    assert cmd[cmd.index("--device") + 1] == "cuda"
    assert cmd[cmd.index("--morpheus-scale") + 1] == "0.0"
    assert cmd[cmd.index("--conditions") + 1] == "age=30"
    assert cmd[cmd.index("--num-samples") + 1] == "10"
    assert calls[0]["cwd"] == repo
    assert calls[0]["check"] is True

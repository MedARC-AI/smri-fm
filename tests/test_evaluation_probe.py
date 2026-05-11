from __future__ import annotations

import json

import pandas as pd
import pytest

from evaluation import DummyBackbone, ProbeConfig, run_evals, run_evals_from_config
from evaluation.tasks.task3_brain_age import BrainAgeTaskConfig


def test_run_evals_task3_probe_with_dummy_model(tmp_path):
    result = run_evals(
        model="dummy",
        profile="probe",
        tasks=["3"],
        output_dir=tmp_path,
        seed=123,
        model_kwargs={"embedding_dim": 4, "input_shape": [1, 4, 4, 4]},
        task_configs={
            "3": BrainAgeTaskConfig(n_train=12, n_validation=6, noise_std=0.0)
        },
        probe_config=ProbeConfig(alpha=0.1),
    )

    run_dir = tmp_path / "eval_probe" / "3__dummy"
    task_dir = run_dir / "task_3_brain_age"
    assert result["run_dir"] == str(run_dir)

    metrics = json.loads((task_dir / "metrics.json").read_text())
    assert set(metrics) == {"mae", "n_train", "n_validation", "pearson_r"}
    assert metrics["n_train"] == 12
    assert metrics["n_validation"] == 6

    predictions = pd.read_csv(task_dir / "predictions.csv")
    assert list(predictions.columns) == ["case_id", "target_age", "predicted_age", "split"]
    assert len(predictions) == 6

    metadata = json.loads((task_dir / "run_metadata.json").read_text())
    assert metadata["profile"] == "probe"
    assert metadata["backbone_trainable_parameters"] == 0
    assert metadata["task_config"]["n_train"] == 12
    assert metadata["probe_config"]["head"] == "ridge"


def test_run_evals_freezes_provided_backbone(tmp_path):
    backbone = DummyBackbone(embedding_dim=4, input_shape=[1, 4, 4, 4])
    backbone.projection.weight.requires_grad_(True)

    run_evals(
        model=backbone,
        profile="probe",
        tasks=["3"],
        output_dir=tmp_path,
        task_configs={"3": {"n_train": 8, "n_validation": 4}},
    )

    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def test_run_evals_from_default_probe_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "output_dir: " + str(tmp_path),
                "model_kwargs:",
                "  embedding_dim: 4",
                "  input_shape: [1, 4, 4, 4]",
                "task_configs:",
                "  '3':",
                "    n_train: 8",
                "    n_validation: 4",
                "probe_config:",
                "  head: ridge",
                "  alpha: 0.5",
            ]
        )
    )

    result = run_evals_from_config(config_path)

    assert "3" in result["tasks"]
    assert (tmp_path / "eval_probe" / "3__dummy" / "config.json").exists()


def test_full_profile_is_not_implemented(tmp_path):
    with pytest.raises(NotImplementedError):
        run_evals(profile="full", output_dir=tmp_path)

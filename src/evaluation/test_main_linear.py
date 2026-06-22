import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset

from evaluation.main_linear import aggregate_folds, main, run_linear
from evaluation.models.registry import register_model
from evaluation.tasks.column import ColumnTask
from evaluation.tasks.registry import register_task


class DummyModel(nn.Module):
    """Echoes the transformed image as its embedding, so features == inputs."""

    def forward(self, batch):
        return batch["feat"]


def dummy_transform(image):
    return {"feat": torch.tensor(image, dtype=torch.float32)}


def test_aggregate_folds_mean_and_std():
    out = aggregate_folds([{"mae": 1.0}, {"mae": 3.0}])
    assert out["mae"] == 2.0
    assert out["mae_std"] == 1.0  # population std of [1, 3]


def _run(task):
    return run_linear(
        task,
        DummyModel(),
        dummy_transform,
        device=torch.device("cpu"),
        batch_size=4,
        num_workers=0,
        seed=0,
    )


def test_run_linear_regression_recovers_linear_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    y = X @ rng.normal(size=5)  # exactly linear => ridge recovers it
    data = Dataset.from_dict(
        {
            "image": X.tolist(),
            "target": y.tolist(),
            "participant_id": [f"p{i}" for i in range(len(y))],
        }
    )
    task = ColumnTask(
        name="reg",
        kind="regression",
        data=data,
        group_column="participant_id",
        n_splits=3,
        seed=0,
    )

    metrics = _run(task)
    assert set(metrics) == {"tput", "summary", "folds"}
    assert len(metrics["folds"]) == 3
    assert metrics["summary"]["mae"] < 0.1


def test_run_linear_classification_handles_string_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y = np.where(X[:, 0] > 0, "pos", "neg")
    X[:, 0] += np.where(y == "pos", 3.0, -3.0)  # make classes clearly separable
    data = Dataset.from_dict(
        {"image": X.tolist(), "target": list(y), "participant_id": [f"p{i}" for i in range(len(y))]}
    )
    task = ColumnTask(
        name="cls",
        kind="classification",
        data=data,
        group_column="participant_id",
        n_splits=3,
        seed=0,
    )

    metrics = _run(task)
    assert "balanced_accuracy" in metrics["summary"]
    assert metrics["summary"]["accuracy"] > 0.8


@register_model
def _linear_test_model():
    return DummyModel(), dummy_transform


@register_task
def _linear_test_task(n_splits: int = 3, seed: int = 0):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    y = X @ rng.normal(size=5)
    data = Dataset.from_dict(
        {
            "image": X.tolist(),
            "target": y.tolist(),
            "participant_id": [f"p{i}" for i in range(len(y))],
        }
    )
    return ColumnTask(
        name="_linear_test_task",
        kind="regression",
        data=data,
        group_column="participant_id",
        n_splits=n_splits,
        seed=seed,
    )


def test_main_passes_task_kwargs(tmp_path):
    metrics = main(
        "_linear_test_model",
        "_linear_test_task",
        overrides=[
            f"output_root={tmp_path}",
            "name=task_kwargs",
            "device=cpu",
            "num_workers=0",
            "task_kwargs.n_splits=4",
            "task_kwargs.seed=17",
        ],
    )
    assert len(metrics["folds"]) == 4
    config = (tmp_path / "task_kwargs" / "config.yaml").read_text()
    assert "n_splits: 4" in config
    assert "seed: 17" in config


def test_main_writes_run_artifacts(tmp_path):
    metrics = main(
        "_linear_test_model",
        "_linear_test_task",
        overrides=[f"output_root={tmp_path}", "device=cpu", "num_workers=0"],
    )
    assert metrics["model"] == "_linear_test_model"

    (summary_path,) = list(tmp_path.rglob("summary.csv"))
    run_dir = summary_path.parent
    assert {"metrics.json", "config.yaml", "log.txt"} <= {p.name for p in run_dir.iterdir()}

    row = pd.read_csv(summary_path)
    assert {"model", "task", "tput"} <= set(row.columns)

import numpy as np
from datasets import Dataset
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from evaluation.tasks.column import ColumnTask


def _data(n=20):
    return Dataset.from_dict(
        {
            "image": [[float(i)] for i in range(n)],
            "age": [40 + i for i in range(n)],
            "sex": ["M", "F"] * (n // 2),
            "participant_id": [f"p{i // 2}" for i in range(n)],
        }
    )


def _task(target="age", kind="regression", n=20):
    return ColumnTask(
        name="t",
        kind=kind,
        data=_data(n),
        group_column="participant_id",
        n_splits=5,
        seed=0,
        target_column=target,
    )


def test_missing_numeric_targets_dropped():
    data = Dataset.from_dict(
        {"image": [1, 2, 3], "y": [1.0, float("nan"), 3.0], "participant_id": ["a", "b", "c"]}
    )
    task = ColumnTask(
        name="t",
        kind="regression",
        data=data,
        group_column="participant_id",
        n_splits=2,
        target_column="y",
    )
    assert len(task.data) == 2


def test_missing_multioutput_targets_dropped():
    data = Dataset.from_dict(
        {
            "image": [1, 2, 3],
            "y": [[1.0, 2.0], [3.0, float("nan")], [5.0, 6.0]],
            "participant_id": ["a", "b", "c"],
        }
    )
    task = ColumnTask(
        name="t",
        kind="regression",
        data=data,
        group_column="participant_id",
        n_splits=2,
        target_column="y",
    )
    assert len(task.data) == 2


def test_missing_string_targets_dropped():
    data = Dataset.from_dict(
        {"image": [1, 2, 3], "y": ["M", None, "F"], "participant_id": ["a", "b", "c"]}
    )
    task = ColumnTask(
        name="t",
        kind="classification",
        data=data,
        group_column="participant_id",
        n_splits=2,
        target_column="y",
    )
    assert len(task.data) == 2


def test_dataset_returns_canonical_columns():
    sample = _task(target="age").dataset()[0]
    assert set(sample) == {"image", "target"}
    assert sample["target"] == 40


def test_split_covers_all_and_is_disjoint():
    folds = list(_task().split())
    assert len(folds) == 5
    assert sum(len(test) for _, test in folds) == 20
    groups = np.asarray(_task().data["participant_id"])
    for train, test in folds:
        assert set(train).isdisjoint(test)
        assert set(groups[train]).isdisjoint(groups[test])


def test_default_splitter_depends_on_kind():
    assert isinstance(_task(kind="regression")._default_splitter(), GroupKFold)
    assert isinstance(
        _task(target="sex", kind="classification")._default_splitter(), StratifiedGroupKFold
    )


def test_metrics_dispatch_on_kind():
    idx = np.arange(4)
    reg = _task(target="age", kind="regression", n=4)
    cls = _task(target="sex", kind="classification", n=4)
    assert set(reg.metrics(np.zeros(4), np.zeros(4), idx)) == {"mae", "rmse", "r2"}
    assert "balanced_accuracy" in cls.metrics(["M", "F", "M", "F"], ["M", "F", "M", "F"], idx)

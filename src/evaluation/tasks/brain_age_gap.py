from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from datasets import Dataset as HFDataset

from evaluation.tasks.base import Kind
from evaluation.tasks.metrics import independent_t_statistic, mae, pearson_r, r2


@dataclass
class BrainAgeGapTask:
    """Train age regression on healthy controls, then score how the brain age
    gap (``age_pred - age_true``) separates cases from controls at test.

    The estimator only ever sees age (``kind="regression"``). The asymmetric
    test objective lives entirely in ``metrics``, which reaches into the
    diagnosis column via ``test_idx``.
    """

    name: str
    data: HFDataset
    age_column: str
    dx_column: str
    control_label: str
    case_label: str
    image_column: str = "image"
    group_column: str | None = None
    test_control_frac: float = 0.2
    seed: int = 0
    kind: Kind = "regression"
    model_selection_eligible: bool = True

    def postprocess_predictions(
        self,
        y_train: np.ndarray,
        train_pred: np.ndarray,
        y_test: np.ndarray,
        test_pred: np.ndarray,
    ) -> np.ndarray:
        slope, intercept = np.polyfit(
            np.asarray(y_train).reshape(-1),
            np.asarray(train_pred).reshape(-1),
            deg=1,
        )
        if np.isclose(slope, 0):
            return test_pred
        return (np.asarray(test_pred) - intercept) / slope

    def dataset(self) -> HFDataset:
        column_mapping = {
            self.image_column: "image",
            self.age_column: "target",
        }
        dataset = self.data.select_columns(list(column_mapping)).rename_columns(column_mapping)
        return dataset

    def split(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        dx = np.asarray(self.data[self.dx_column])
        controls = np.where(dx == self.control_label)[0]
        cases = np.where(dx == self.case_label)[0]

        # hold out some controls so the test set is leakage-free
        rng = np.random.default_rng(self.seed)
        if self.group_column is not None:
            # split by subject so a subject's sessions never straddle train/test
            groups = np.asarray(self.data[self.group_column])
            subjects = rng.permutation(np.unique(groups[controls]))
            n_test = round(self.test_control_frac * len(subjects))
            test_subjects = set(subjects[:n_test])
            in_test = np.isin(groups[controls], list(test_subjects))
            test_controls, train_controls = controls[in_test], controls[~in_test]
        else:
            controls = rng.permutation(controls)
            n_test = round(self.test_control_frac * len(controls))
            test_controls, train_controls = controls[:n_test], controls[n_test:]

        yield train_controls, np.concatenate([test_controls, cases])

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        test_idx: np.ndarray,
        y_score: np.ndarray | None = None,
    ) -> dict:
        gap = (np.asarray(y_pred) - np.asarray(y_true)).reshape(-1)
        diagnoses = np.asarray(self.data[self.dx_column])[test_idx]
        return {
            "pearson_r": pearson_r(y_true, y_pred),
            "r2": r2(y_true, y_pred),
            "mae_years_bias_corrected": mae(y_true, y_pred),
            "bag_tstat": independent_t_statistic(
                gap[diagnoses == self.case_label],
                gap[diagnoses == self.control_label],
            ),
        }

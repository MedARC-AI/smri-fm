from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from torch import Tensor, nn

from evaluation.config_schema import ProbeConfig, TaskConfig
from evaluation.tasks.base import EvalTask


@dataclass(frozen=True)
class BrainAgeDataset:
    images: Tensor
    ages: np.ndarray
    case_ids: list[str]


@dataclass(frozen=True)
class BrainAgeTaskConfig(TaskConfig):
    source: str = "synthetic"
    n_train: int = 32
    n_validation: int = 16
    noise_std: float = 0.1

    @classmethod
    def from_mapping(
        cls,
        raw_config: dict[str, Any] | TaskConfig | "BrainAgeTaskConfig" | None,
    ) -> "BrainAgeTaskConfig":
        if raw_config is None:
            return cls()
        if isinstance(raw_config, cls):
            return raw_config

        config = raw_config.to_dict() if isinstance(raw_config, TaskConfig) else dict(raw_config)
        kwargs = dict(config.pop("kwargs", {}))
        source = str(config.pop("source", cls.source))
        n_train = int(config.pop("n_train", kwargs.pop("n_train", cls.n_train)))
        n_validation = int(
            config.pop("n_validation", kwargs.pop("n_validation", cls.n_validation))
        )
        noise_std = float(config.pop("noise_std", kwargs.pop("noise_std", cls.noise_std)))
        kwargs.update(config)
        return cls(
            source=source,
            n_train=n_train,
            n_validation=n_validation,
            noise_std=noise_std,
            kwargs=kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "n_train": self.n_train,
                "n_validation": self.n_validation,
                "noise_std": self.noise_std,
            }
        )
        return payload


class BrainAgeTask(EvalTask[BrainAgeTaskConfig]):
    id = "3"
    name = "brain_age"
    output_name = "task_3_brain_age"
    config_type = BrainAgeTaskConfig

    def run_probe(
        self,
        backbone: nn.Module,
        output_dir: Path,
        data_dir: Path,
        seed: int,
        device: str,
        task_config: BrainAgeTaskConfig,
        probe_config: ProbeConfig,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)

        train, validation = self._load_data(
            data_dir=data_dir,
            seed=seed,
            task_config=task_config,
            input_shape=_infer_input_shape(backbone),
        )

        backbone.to(device)
        backbone.eval()
        with torch.no_grad():
            train_embeddings = _encode(backbone, train.images, device=device)
            validation_embeddings = _encode(backbone, validation.images, device=device)

        if probe_config.head != "ridge":
            raise NotImplementedError("Task 3 probe currently supports head='ridge'.")

        probe = Ridge(alpha=probe_config.alpha, **dict(probe_config.kwargs))
        probe.fit(train_embeddings, train.ages)
        predictions = probe.predict(validation_embeddings)

        metrics = {
            "mae": float(mean_absolute_error(validation.ages, predictions)),
            "pearson_r": _pearsonr(validation.ages, predictions),
            "n_train": len(train.ages),
            "n_validation": len(validation.ages),
        }

        predictions_df = pd.DataFrame(
            {
                "case_id": validation.case_ids,
                "target_age": validation.ages,
                "predicted_age": predictions,
                "split": "validation",
            }
        )
        predictions_df.to_csv(output_dir / "predictions.csv", index=False)
        _write_json(output_dir / "metrics.json", metrics)
        _write_json(
            output_dir / "run_metadata.json",
            {
                "task_id": self.id,
                "task_name": self.name,
                "profile": "probe",
                "backbone_trainable_parameters": _count_trainable_parameters(backbone),
                "probe_config": probe_config.to_dict(),
                "task_config": task_config.to_dict(),
            },
        )

        return {
            "task_id": self.id,
            "name": self.name,
            "output_dir": str(output_dir),
            "metrics": metrics,
        }

    def _load_data(
        self,
        data_dir: Path,
        seed: int,
        task_config: BrainAgeTaskConfig,
        input_shape: tuple[int, ...],
    ) -> tuple[BrainAgeDataset, BrainAgeDataset]:
        if task_config.source != "synthetic":
            raise NotImplementedError(
                "Task 3 currently supports source='synthetic'. "
                f"Got source={task_config.source!r}; "
                "add a loader here when local FOMO26 files are available."
            )
        return _make_synthetic_brain_age_data(
            input_shape=input_shape,
            n_train=task_config.n_train,
            n_validation=task_config.n_validation,
            noise_std=task_config.noise_std,
            seed=seed,
        )


def _make_synthetic_brain_age_data(
    input_shape: tuple[int, ...],
    n_train: int,
    n_validation: int,
    noise_std: float,
    seed: int,
) -> tuple[BrainAgeDataset, BrainAgeDataset]:
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_total = n_train + n_validation
    images = torch.randn((n_total, *input_shape), generator=generator, dtype=torch.float32)

    signal = images.flatten(start_dim=1)[:, : min(32, images[0].numel())].mean(dim=1).numpy()
    ages = 55.0 + 12.0 * signal + rng.normal(0.0, noise_std, size=n_total)

    train = BrainAgeDataset(
        images=images[:n_train],
        ages=ages[:n_train],
        case_ids=[f"synthetic_train_{idx:04d}" for idx in range(n_train)],
    )
    validation = BrainAgeDataset(
        images=images[n_train:],
        ages=ages[n_train:],
        case_ids=[f"synthetic_validation_{idx:04d}" for idx in range(n_validation)],
    )
    return train, validation


def _infer_input_shape(backbone: nn.Module) -> tuple[int, ...]:
    input_shape = getattr(backbone, "input_shape", None)
    if input_shape is None:
        raise ValueError(
            "Task 3 synthetic data needs the backbone to expose an input_shape attribute."
        )
    return tuple(input_shape)


def _encode(backbone: nn.Module, images: Tensor, device: str) -> np.ndarray:
    encode = getattr(backbone, "encode", backbone)
    embeddings = encode(images.to(device))
    return embeddings.detach().cpu().numpy()


def _pearsonr(targets: np.ndarray, predictions: np.ndarray) -> float:
    if len(targets) < 2:
        return float("nan")
    corr = np.corrcoef(targets, predictions)[0, 1]
    return float(corr)


def _count_trainable_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

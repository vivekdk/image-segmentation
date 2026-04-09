from __future__ import annotations

"""Configuration dataclasses and YAML loading helpers for experiments.

The project keeps configuration intentionally small and explicit so that a run
can be reproduced from a single YAML file saved alongside checkpoints. The
dataclasses act as a typed contract between the CLI entrypoints and the rest of
the training stack.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    dataset_name: str = "oxford_pet"
    root_dir: str = "data"
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 0
    val_fraction: float = 0.15
    seed: int = 42
    download: bool = True


@dataclass(slots=True)
class ModelConfig:
    in_channels: int = 3
    out_channels: int = 1
    base_channels: int = 32


@dataclass(slots=True)
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 25


@dataclass(slots=True)
class TrainingConfig:
    device: str = "auto"
    threshold: float = 0.5
    early_stopping_patience: int = 5
    log_every_n_steps: int = 10
    output_root: str = "outputs"
    run_name: str = "oxford_pet_unet"
    save_visualizations: bool = True
    num_visualization_samples: int = 4


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(data_cls: type[Any], raw: dict[str, Any] | None) -> Any:
    """Instantiate a config dataclass from a possibly-missing YAML section."""
    raw = raw or {}
    return data_cls(**raw)


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load a YAML config file into the typed experiment configuration."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return ExperimentConfig(
        data=_merge_dataclass(DataConfig, raw.get("data")),
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        optimizer=_merge_dataclass(OptimizerConfig, raw.get("optimizer")),
        training=_merge_dataclass(TrainingConfig, raw.get("training")),
    )


def save_config(config: ExperimentConfig, output_path: str | Path) -> None:
    """Persist the resolved config next to run artifacts for reproducibility."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)

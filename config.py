"""Experiment configuration for BUAL on CIFAR-10.

The module exposes a typed configuration object for new code while keeping the
legacy constant names used by earlier scripts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ExperimentConfig:
    """Centralized, immutable training configuration."""

    num_classes: int = 10
    num_train: int = 50_000
    batch_size: int = 128
    subset_size: int = 10_000
    acquisition_size: int = 1_000

    trials: int = 5
    cycles: int = 10
    epochs: int = 200
    loss_warmup_epochs: int = 120

    learning_rate: float = 0.1
    milestones: Tuple[int, ...] = (160,)
    momentum: float = 0.9
    weight_decay: float = 5e-4

    seed: int = 42
    data_dir: Path = Path("C:/data")
    output_dir: Path = Path("Log") / "CIFAR10"
    num_workers: int = 0
    pin_memory: bool = True

    @property
    def num_val(self) -> int:
        return 50_000 - self.num_train


DEFAULT_CONFIG = ExperimentConfig()

# Backward-compatible constants.
NUM_CLASS = DEFAULT_CONFIG.num_classes
NUM_TRAIN = DEFAULT_CONFIG.num_train
NUM_VAL = DEFAULT_CONFIG.num_val
BATCH = DEFAULT_CONFIG.batch_size
SUBSET = DEFAULT_CONFIG.subset_size
ADDENDUM = DEFAULT_CONFIG.acquisition_size

TRIALS = DEFAULT_CONFIG.trials
CYCLES = DEFAULT_CONFIG.cycles

EPOCH = DEFAULT_CONFIG.epochs
LR = DEFAULT_CONFIG.learning_rate
MILESTONES = list(DEFAULT_CONFIG.milestones)
EPOCHL = DEFAULT_CONFIG.loss_warmup_epochs

MOMENTUM = DEFAULT_CONFIG.momentum
WDECAY = DEFAULT_CONFIG.weight_decay

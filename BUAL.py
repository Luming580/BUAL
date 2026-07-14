import argparse
import os
import random
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import CIFAR10

import model.lossnet as lossnet
from config import DEFAULT_CONFIG, ExperimentConfig
from model.resnet import ResNet18
from sampler import SubsetSequentialSampler
from utils import ensure_dir, get_sigmoid_alpha, min_max_normalize, train, train_wwt


ModelBundle = Dict[str, torch.nn.Module]


def seed_everything(seed: int = 42, deterministic: bool = True):
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms() -> Tuple[T.Compose, T.Compose]:
    train_transform = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )
    test_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )
    return train_transform, test_transform


def build_datasets(config: ExperimentConfig) -> Tuple[CIFAR10, CIFAR10, CIFAR10]:
    train_transform, test_transform = build_transforms()
    data_dir = str(config.data_dir)
    train_dataset = CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    unlabeled_dataset = CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    return train_dataset, unlabeled_dataset, test_dataset


def loader_kwargs(config: ExperimentConfig, device: torch.device) -> Dict[str, object]:
    return {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory and device.type == "cuda",
    }


def build_dataloaders(
    config: ExperimentConfig,
    device: torch.device,
    train_dataset: CIFAR10,
    test_dataset: CIFAR10,
    labeled_indices: Sequence[int],
) -> Dict[str, DataLoader]:
    kwargs = loader_kwargs(config, device)
    return {
        "train": DataLoader(
            train_dataset,
            sampler=SubsetRandomSampler(labeled_indices),
            **kwargs,
        ),
        "test": DataLoader(test_dataset, shuffle=False, **kwargs),
    }


def build_models(config: ExperimentConfig, device: torch.device) -> ModelBundle:
    return {
        "backbone": ResNet18(config.num_classes).to(device),
        "dwt": ResNet18(config.num_classes, flag=True).to(device),
        "module": lossnet.LossNet().to(device),
    }


def build_optimizers(
    models: Mapping[str, torch.nn.Module],
    config: ExperimentConfig,
) -> Dict[str, optim.Optimizer]:
    optimizer_kwargs = {
        "lr": config.learning_rate,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
    }
    return {
        "backbone": optim.SGD(models["backbone"].parameters(), **optimizer_kwargs),
        "dwt": optim.SGD(models["dwt"].parameters(), **optimizer_kwargs),
        "module": optim.SGD(models["module"].parameters(), **optimizer_kwargs),
    }


def build_schedulers(
    optimizers: Mapping[str, optim.Optimizer],
    config: ExperimentConfig,
) -> Dict[str, lr_scheduler.MultiStepLR]:
    return {
        name: lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(config.milestones),
            gamma=0.1,
        )
        for name, optimizer in optimizers.items()
    }


@torch.no_grad()
def get_uncertainty(
    models: Mapping[str, torch.nn.Module],
    unlabeled_loader: DataLoader,
    alpha: float = 0.8,
) -> torch.Tensor:
    """Combine branch disagreement and predicted loss into one uncertainty score."""

    models["backbone"].eval()
    models["dwt"].eval()
    models["module"].eval()

    discrepancy_list = []
    predicted_loss_list = []
    device = next(models["backbone"].parameters()).device

    for inputs, _ in unlabeled_loader:
        inputs = inputs.to(device, non_blocking=True)

        out_bb, _, _, features = models["backbone"](inputs)
        prob_bb = F.softmax(out_bb, dim=1, dtype=torch.float32)

        pred_loss = models["module"](features).view(inputs.size(0))

        out_dwt, _, _, _ = models["dwt"](inputs)
        prob_dwt = F.softmax(out_dwt, dim=1, dtype=torch.float32)

        discrepancy = 1 - F.cosine_similarity(prob_bb, prob_dwt, dim=1)
        discrepancy_list.append(discrepancy.cpu())
        predicted_loss_list.append(pred_loss.cpu())

    if not discrepancy_list:
        return torch.empty(0)

    discrepancy_tensor = torch.cat(discrepancy_list, dim=0)
    pred_loss_tensor = torch.cat(predicted_loss_list, dim=0)

    u_discrepancy = min_max_normalize(discrepancy_tensor, mode="rank")
    u_loss = min_max_normalize(pred_loss_tensor, mode="rank")
    return alpha * u_discrepancy + (1 - alpha) * u_loss


def acquire_samples(
    unlabeled_set: Sequence[int],
    subset: Sequence[int],
    uncertainty: torch.Tensor,
    acquisition_size: int,
) -> Tuple[List[int], List[int]]:
    if uncertainty.numel() != len(subset):
        raise ValueError("Uncertainty score count must match the unlabeled subset size.")

    query_size = min(acquisition_size, len(subset))
    if query_size == 0:
        return [], list(unlabeled_set)

    subset_tensor = torch.as_tensor(subset)
    ranking = torch.argsort(uncertainty)
    acquired = subset_tensor[ranking[-query_size:]].tolist()
    retained_subset = subset_tensor[ranking[:-query_size]].tolist()
    retained_pool = retained_subset + list(unlabeled_set[len(subset) :])
    return acquired, retained_pool


def log_line(message: str, log_file) -> None:
    print(message, flush=True)
    log_file.write(message + "\n")
    log_file.flush()


def run_experiment(config: ExperimentConfig = DEFAULT_CONFIG) -> None:
    seed_everything(config.seed)
    device = resolve_device()

    output_dir = ensure_dir(config.output_dir)
    log_path = output_dir / "Log.txt"
    train_dataset, unlabeled_dataset, test_dataset = build_datasets(config)

    with log_path.open("w", encoding="utf-8") as log_file:
        log_line(f">> Device: {device}", log_file)
        log_line(f">> Output directory: {output_dir}", log_file)

        for trial in range(config.trials):
            indices = list(range(config.num_train))
            random.shuffle(indices)
            labeled_set = indices[: config.acquisition_size]
            unlabeled_set = indices[config.acquisition_size :]

            dataloaders = build_dataloaders(
                config,
                device,
                train_dataset,
                test_dataset,
                labeled_set,
            )
            models = build_models(config, device)

            for cycle in range(config.cycles):
                log_line("################################################", log_file)

                criterion = nn.CrossEntropyLoss(reduction="none")
                optimizers = build_optimizers(models, config)
                schedulers = build_schedulers(optimizers, config)

                acc = train(models, criterion, optimizers, schedulers, dataloaders, config.epochs, cycle, config.loss_warmup_epochs, log_file, device=device)
                log_line((
                        f"Trial {trial + 1}/{config.trials} || "
                        f"Cycle {cycle + 1}/{config.cycles} || "
                        f"Label set size {len(labeled_set)}: Test acc {acc}"
                    ), log_file, )

                train_wwt(models,criterion,optimizers,schedulers,dataloaders,config.epochs, device=device,)

                random.shuffle(unlabeled_set)
                subset_size = min(config.subset_size, len(unlabeled_set))
                subset = unlabeled_set[:subset_size]
                unlabeled_loader = DataLoader(
                    unlabeled_dataset,
                    batch_size=config.batch_size,
                    sampler=SubsetSequentialSampler(subset),
                    num_workers=config.num_workers,
                    pin_memory=config.pin_memory and device.type == "cuda",
                )

                alpha_adaptive = get_sigmoid_alpha(cycle, shift=6.0)
                log_line(f"Cycle {cycle}: Sigmoid Alpha = {alpha_adaptive:.4f}", log_file)

                uncertainty = get_uncertainty(models, unlabeled_loader, alpha=alpha_adaptive)
                acquired, unlabeled_set = acquire_samples( unlabeled_set, subset,uncertainty,config.acquisition_size)
                labeled_set.extend(acquired)

                dataloaders["train"] = build_dataloaders(config,device,train_dataset,test_dataset,labeled_set)["train"]

        log_line("---------------------------Current Trial is done-----------------------------", log_file)


def BUAL(config: ExperimentConfig = DEFAULT_CONFIG) -> None:
    """Legacy entry point."""
    run_experiment(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BUAL CIFAR-10 experiment.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_CONFIG.data_dir)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--trials", type=int, default=DEFAULT_CONFIG.trials)
    parser.add_argument("--cycles", type=int, default=DEFAULT_CONFIG.cycles)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--subset-size", type=int, default=DEFAULT_CONFIG.subset_size)
    parser.add_argument("--acquisition-size", type=int, default=DEFAULT_CONFIG.acquisition_size)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG.num_workers)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return replace(
        DEFAULT_CONFIG,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        trials=args.trials,
        cycles=args.cycles,
        epochs=args.epochs,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        acquisition_size=args.acquisition_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    BUAL(config_from_args(parse_args()))
    print("BUAL experiment is done.")

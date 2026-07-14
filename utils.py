"""Training, evaluation, and experiment utilities for BUAL."""

from pathlib import Path
from typing import IO, Mapping, MutableMapping, Optional
import math
import shutil

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from sampler import SubsetSequentialSampler

iters = 0


def _infer_device(model):
    return next(model.parameters()).device


def _write_log(message, log_file):
    print(message, flush=True)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


def loss_prediction_loss(input_tensor, target, margin=1.0, reduction="mean"):
    if len(input_tensor) % 2 != 0:
        raise ValueError("loss_prediction_loss expects an even batch size.")
    if input_tensor.shape != input_tensor.flip(0).shape:
        raise ValueError("input_tensor must be compatible with flipped pairing.")

    input_tensor = (input_tensor - input_tensor.flip(0))[: len(input_tensor) // 2]
    target = (target - target.flip(0))[: len(target) // 2].detach()
    signs = 2 * torch.sign(torch.clamp(target, min=0)) - 1
    losses = torch.clamp(margin - signs * input_tensor, min=0)

    if reduction == "mean":
        return losses.mean()
    if reduction == "none":
        return losses
    raise NotImplementedError(f"Unsupported reduction: {reduction}")


LossPredLoss = loss_prediction_loss


def evaluate(model, dataloaders, mode, device):
    if mode not in dataloaders:
        raise KeyError(f"Unknown dataloader split: {mode}")

    model.eval()
    device = device or _infer_device(model)

    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloaders[mode]:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            scores = model(inputs)[0]

            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total if total else 0.0


def train_epoch_wwt(models, criterion, optimizers, dataloaders, device):
    models["dwt"].train()
    device = device or _infer_device(models["dwt"])
    global iters

    for inputs, labels in dataloaders["train"]:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        iters += 1

        optimizers["dwt"].zero_grad()

        scores, _, _, _ = models["dwt"](inputs)
        target_loss = criterion(scores, labels)
        loss = target_loss.mean()

        loss.backward()
        optimizers["dwt"].step()


def train_wwt(models, criterion, optimizers, schedulers, dataloaders, num_epochs, device):
    for _ in range(num_epochs):
        train_epoch_wwt(models, criterion, optimizers, dataloaders, device=device)
        schedulers["dwt"].step()


def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, device):
    models["backbone"].train()
    models["module"].train()
    device = device or _infer_device(models["backbone"])
    global iters

    for inputs, labels in dataloaders["train"]:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        iters += 1

        optimizers["backbone"].zero_grad()
        optimizers["module"].zero_grad()

        scores, _, _, features = models["backbone"](inputs)
        target_loss = criterion(scores, labels)
        task_loss = target_loss.mean()

        if epoch > epoch_loss:
            features = [feature.detach() for feature in features]

        pred_loss = models["module"](features).view(target_loss.size(0))
        module_loss = loss_prediction_loss(pred_loss, target_loss)
        loss = task_loss + 0.8 * module_loss

        loss.backward()
        optimizers["backbone"].step()
        optimizers["module"].step()


def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, cycle, epoch_loss, log_file, device):
    _write_log(">> Training the model...", log_file)
    best_acc = 0.0
    for epoch in range(num_epochs):
        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, device=device)
        schedulers["backbone"].step()
        schedulers["module"].step()

        if epoch % 20 == 0 or epoch == num_epochs - 1:
            acc = evaluate(models["backbone"], dataloaders, "test", device=device)
            best_acc = max(best_acc, acc)
            _write_log(
                (
                    f"Cycle:{cycle} Epoch:{epoch} --- "
                    f"Val Acc:{acc:.3f} Best Acc:{best_acc:.3f}"
                ), log_file)

    _write_log(">> Finished.", log_file)
    return best_acc


def min_max_normalize(tensor, mode, **legacy_kwargs):
    if "type" in legacy_kwargs:
        mode = legacy_kwargs.pop("type")
    if legacy_kwargs:
        unexpected = ", ".join(legacy_kwargs)
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    if tensor.numel() <= 1:
        return torch.zeros_like(tensor, dtype=torch.float32)

    if mode == "value":
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        return (tensor - min_val) / (max_val - min_val + 1e-8)

    sorted_idx = torch.argsort(tensor)
    ranks = torch.zeros_like(tensor, dtype=torch.float32, device=tensor.device)
    ranks[sorted_idx] = torch.arange(len(tensor), dtype=torch.float32, device=tensor.device)
    return ranks / (len(tensor) - 1)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def CreateFolder(image_folder):
    return create_clean_dir(Path(image_folder))


def get_sigmoid_alpha(cycle, shift):
    return 1.0 / (1.0 + math.exp(-0.5 * (cycle - shift)))


class DistributionLogger:
    """Record and plot feature-distribution statistics across AL cycles."""

    def __init__(self, save_dir, trial: int = 0):
        self.save_dir = ensure_dir(Path(save_dir))
        self.trial = trial
        self.history = {
            "cycle": [],
            "mean_norm": [],
            "mean_diff": [],
            "var_avg": [],
            "feature_norms": {},
        }
        self.prev_mean = None

    def record(self, models, dataset, labeled_indices, cycle: int) -> None:
        models["backbone"].eval()
        device = _infer_device(models["backbone"])

        loader = DataLoader(
            dataset,
            batch_size=128,
            sampler=SubsetSequentialSampler(labeled_indices),
            num_workers=0,
        )

        all_feats = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                _, _, _, feats_list = models["backbone"](inputs)
                feat = feats_list[-1]

                if len(feat.shape) > 2:
                    feat = torch.mean(feat, dim=[2, 3])

                all_feats.append(feat.cpu())

        features = torch.cat(all_feats, dim=0)

        curr_mean = torch.mean(features, dim=0)
        curr_var = torch.var(features, dim=0).mean().item()
        norms = torch.norm(features, dim=1).numpy()

        if self.prev_mean is not None:
            diff = torch.norm(curr_mean - self.prev_mean, p=2).item()
        else:
            diff = 0.0

        self.prev_mean = curr_mean

        self.history["cycle"].append(cycle)
        self.history["mean_norm"].append(torch.norm(curr_mean).item())
        self.history["mean_diff"].append(diff)
        self.history["var_avg"].append(curr_var)
        self.history["feature_norms"][cycle] = norms

        np.save(self.save_dir / f"stats_history{self.trial}.npy", self.history)

        print(f">> [Stats Recorded] Cycle {cycle}: Mean Shift={diff:.4f}, Var={curr_var:.4f}")

    def plot_final(self) -> None:
        print(">> Generating Final Plots...")

        cycles = self.history["cycle"]
        if not cycles:
            return

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = "tab:red"
        ax1.set_xlabel("Active Learning Cycle", fontsize=12)
        ax1.set_ylabel("Mean Shift ||mu_t - mu_{t-1}||", color=color, fontsize=12)
        ax1.plot(
            cycles,
            self.history["mean_diff"],
            "o-",
            color=color,
            linewidth=2,
            label="Mean Stability",
        )
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("Feature Variance (Avg)", color=color, fontsize=12)
        ax2.plot(
            cycles,
            self.history["var_avg"],
            "s--",
            color=color,
            linewidth=2,
            label="Variance",
        )
        ax2.tick_params(axis="y", labelcolor=color)

        plt.title("Convergence of Feature Distribution Statistics", fontsize=14)
        fig.tight_layout()
        plt.savefig(self.save_dir / "Final_Stability_Convergence.png", dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))

        selected_indices = range(min(10, len(cycles)))
        palette = sns.color_palette("viridis", n_colors=len(list(selected_indices)))

        for i, idx in enumerate(range(min(10, len(cycles)))):
            cycle = cycles[idx]
            if cycle in self.history["feature_norms"]:
                data = self.history["feature_norms"][cycle]
                sns.kdeplot(
                    data,
                    fill=True,
                    color=palette[i],
                    alpha=0.3,
                    linewidth=2,
                    label=f"Cycle {cycle} (N={len(data)})",
                )

        plt.title("Evolution of Feature Norm Distribution", fontsize=14)
        plt.xlabel("Feature L2 Norm", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / "Final_Distribution_Evolution.png", dpi=300)
        plt.close()

        print(f">> Plots saved to {self.save_dir}")

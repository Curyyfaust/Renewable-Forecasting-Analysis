"""Training utilities for provincial 24-hour wind-power forecasting."""

from __future__ import annotations

import argparse
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dataloader import WindForecastDataset
from model import MODEL_REGISTRY, WindForecastModel, build_model


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    history = batch["history_wind"].to(device, non_blocking=True)
    weather = batch["future_weather"].to(device, non_blocking=True)
    target = batch["target_wind"].to(device, non_blocking=True)
    wind_mean = batch["wind_mean"].to(device, non_blocking=True)
    wind_std = batch["wind_std"].to(device, non_blocking=True)
    return history, weather, target, wind_mean, wind_std


class NormalizedWindDataset(Dataset):
    """View of selected samples normalized with training-only statistics."""

    def __init__(
        self,
        dataset: WindForecastDataset,
        indices: range,
        wind_mean: np.ndarray,
        wind_std: np.ndarray,
        weather_mean: np.ndarray,
        weather_std: np.ndarray,
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.wind_mean = torch.as_tensor(wind_mean, dtype=torch.float32)
        self.wind_std = torch.as_tensor(wind_std, dtype=torch.float32)
        self.weather_mean = torch.as_tensor(weather_mean, dtype=torch.float32)
        self.weather_std = torch.as_tensor(weather_std, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[self.indices[index]]
        return {
            "history_wind":
                (sample["history_wind"] - self.wind_mean) / self.wind_std,
            "future_weather":
                (sample["future_weather"] - self.weather_mean) / self.weather_std,
            "target_wind":
                (sample["target_wind"] - self.wind_mean) / self.wind_std,
            "wind_mean": self.wind_mean,
            "wind_std": self.wind_std,
            "origin_ns": sample["origin_ns"],
        }


def train_one_epoch(
    model: WindForecastModel,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    max_grad_norm: float | None = 1.0,
) -> dict[str, float]:
    """Train for one epoch and return sample-weighted MSE and MAE."""
    model.train()
    total_loss = 0.0
    total_absolute_error = 0.0
    total_elements = 0

    for batch in loader:
        history, weather, target, wind_mean, wind_std = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=scaler.is_enabled(),
        ):
            prediction = model(history, weather)
            loss = criterion(prediction, target)

        scaler.scale(loss).backward()
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        element_count = target.numel()
        wind_mean = wind_mean.unsqueeze(1)
        wind_std = wind_std.unsqueeze(1)
        prediction_mw = prediction.detach() * wind_std + wind_mean
        target_mw = target * wind_std + wind_mean
        total_loss += (prediction_mw - target_mw).square().sum().item()
        total_absolute_error += (prediction_mw - target_mw).abs().sum().item()
        total_elements += element_count

    if total_elements == 0:
        raise ValueError("Training loader is empty")
    return {
        "mse": total_loss / total_elements,
        "mae": total_absolute_error / total_elements,
    }


@torch.inference_mode()
def evaluate(
    model: WindForecastModel,
    loader: Iterable[dict[str, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a model without updating parameters."""
    model.eval()
    total_loss = 0.0
    total_absolute_error = 0.0
    total_elements = 0

    for batch in loader:
        history, weather, target, wind_mean, wind_std = _move_batch(batch, device)
        prediction = model(history, weather)
        loss = criterion(prediction, target)
        element_count = target.numel()
        wind_mean = wind_mean.unsqueeze(1)
        wind_std = wind_std.unsqueeze(1)
        prediction_mw = prediction * wind_std + wind_mean
        target_mw = target * wind_std + wind_mean
        total_loss += (prediction_mw - target_mw).square().sum().item()
        total_absolute_error += (prediction_mw - target_mw).abs().sum().item()
        total_elements += element_count

    if total_elements == 0:
        raise ValueError("Validation loader is empty")
    return {
        "mse": total_loss / total_elements,
        "mae": total_absolute_error / total_elements,
    }


def train_model(
    model: WindForecastModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    max_grad_norm: float | None = 1.0,
    device: str | torch.device | None = None,
    checkpoint_path: str | Path | None = None,
) -> list[dict[str, float]]:
    """Train a model and optionally save the best validation checkpoint.

    The returned list contains one metric dictionary per epoch. If no
    validation loader is supplied, training MSE determines the best epoch.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(selected_device)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=selected_device.type == "cuda"
    )

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float]] = []
    best_score = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            selected_device,
            scaler,
            max_grad_norm,
        )
        metrics: dict[str, float] = {
            "epoch": float(epoch),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_mse": train_metrics["mse"],
            "train_mae": train_metrics["mae"],
        }

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, selected_device)
            metrics.update(
                val_mse=val_metrics["mse"], val_mae=val_metrics["mae"]
            )
            score = val_metrics["mse"]
        else:
            score = train_metrics["mse"]

        scheduler.step(score)
        history.append(metrics)
        message = (
            f"epoch {epoch:03d}/{epochs:03d} "
            f"train_mse={metrics['train_mse']:.6f} "
            f"train_mae={metrics['train_mae']:.6f}"
        )
        if val_loader is not None:
            message += (
                f" val_mse={metrics['val_mse']:.6f}"
                f" val_mae={metrics['val_mae']:.6f}"
            )
        print(message, flush=True)

        if score < best_score:
            best_score = score
            if checkpoint is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "score": score,
                        "metrics": metrics,
                    },
                    checkpoint,
                )

    return history


def make_2019_loaders(
    batch_size: int = 32,
    validation_fraction: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create deterministic 2019 train/validation loaders.

    The split is chronological: the first portion trains the model and the
    final portion validates it. No data outside 2019 is used.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    dataset = WindForecastDataset(year=2019)
    validation_size = max(1, round(len(dataset) * validation_fraction))
    split_index = len(dataset) - validation_size
    # Statistics use only timestamps and forecasts visible to the training
    # portion. Per-province wind scaling preserves regional capacity levels;
    # weather scaling is per meteorological feature.
    last_training_hour = int(dataset.origin_hour_offsets[split_index - 1]) + 24
    training_wind = dataset.wind[: last_training_hour + 1]
    wind_mean = training_wind.mean(axis=0)
    wind_std = training_wind.std(axis=0)
    weather_mean = dataset.forecast[:split_index].mean(axis=(0, 1, 2))
    weather_std = dataset.forecast[:split_index].std(axis=(0, 1, 2))
    wind_std = np.maximum(wind_std, 1e-6)
    weather_std = np.maximum(weather_std, 1e-6)

    train_dataset = NormalizedWindDataset(
        dataset,
        range(0, split_index),
        wind_mean,
        wind_std,
        weather_mean,
        weather_std,
    )
    validation_dataset = NormalizedWindDataset(
        dataset,
        range(split_index, len(dataset)),
        wind_mean,
        wind_std,
        weather_mean,
        weather_std,
    )

    generator = torch.Generator().manual_seed(seed)
    common: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **common
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
    return train_loader, validation_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default="dnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_loader, validation_loader = make_2019_loaders(
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model = build_model(args.model)
    train_model(
        model,
        train_loader,
        validation_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()

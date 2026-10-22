"""Train and compare all four wind-power forecasting models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from exp import make_2019_loaders, set_seed, train_model
from model import MODEL_REGISTRY, build_model


DEFAULT_MODELS = ("cnn", "lstm", "transformer", "dnn")


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    """Write one model's epoch metrics as CSV."""
    if not history:
        raise ValueError("Training history is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(history[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)


def train_all_models(
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    validation_fraction: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
    device: str | None = None,
    output_dir: str | Path = "results",
) -> dict[str, dict[str, float | int | str]]:
    """Train the requested models sequentially on the same 2019 split."""
    unknown = sorted(set(models).difference(MODEL_REGISTRY))
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}")

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, float | int | str]] = {}

    print(f"device={selected_device}")
    print(f"models={', '.join(models)}")
    print(f"epochs={epochs} batch_size={batch_size}")

    for position, model_name in enumerate(models, start=1):
        print(f"\n[{position}/{len(models)}] training {model_name}", flush=True)

        # Recreate loaders and reset the seed so every model sees the same
        # chronological split and the same training shuffle order.
        set_seed(seed)
        train_loader, validation_loader = make_2019_loaders(
            batch_size=batch_size,
            validation_fraction=validation_fraction,
            num_workers=num_workers,
            seed=seed,
        )
        model = build_model(model_name)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        checkpoint_path = destination / f"{model_name}_best.pt"

        history = train_model(
            model,
            train_loader,
            validation_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=selected_device,
            checkpoint_path=checkpoint_path,
        )
        history_path = destination / f"{model_name}_history.csv"
        write_history(history_path, history)

        best_epoch_metrics = min(history, key=lambda row: row["val_mse"])
        summary[model_name] = {
            "parameters": parameter_count,
            "best_epoch": int(best_epoch_metrics["epoch"]),
            "best_val_mse": best_epoch_metrics["val_mse"],
            "best_val_mae": best_epoch_metrics["val_mae"],
            "checkpoint": str(checkpoint_path),
            "history": str(history_path),
        }
        print(
            f"{model_name} complete: best_epoch="
            f"{summary[model_name]['best_epoch']} "
            f"best_val_mse={summary[model_name]['best_val_mse']:.6f}",
            flush=True,
        )

        del model, train_loader, validation_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = destination / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"\nsummary saved to {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_REGISTRY),
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_all_models(
        models=tuple(args.models),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

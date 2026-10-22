"""Generate daily 24-hour wind forecasts for 2020 and a 1x4 comparison figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from exp import make_2019_loaders
from model import MODEL_REGISTRY, build_model


PROJECT_ROOT = Path(__file__).resolve().parent
WIND_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "provincial_hourly_2018_2023"
    / "provincial_hourly_wind_2018_2023.csv"
)
WEATHER_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "weather_forecast_24h"
    / "weatherbench2_hres_china_2018_2022.npz"
)
MODEL_NAMES = ("cnn", "lstm", "transformer", "dnn")


def load_2020_daily_inputs(
    issue_hour: int = 8,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    """Build 366 daily samples without discarding year-boundary context."""
    wind_frame = pd.read_csv(WIND_PATH, parse_dates=["timestamp_beijing"])
    wind_frame = wind_frame.set_index("timestamp_beijing").sort_index()
    archive = np.load(WEATHER_PATH, allow_pickle=False)
    origins_all = pd.DatetimeIndex(archive["origin"])
    provinces = [str(value) for value in archive["provinces"].tolist()]

    selected = np.flatnonzero(
        (origins_all.year == 2020) & (origins_all.hour == issue_hour)
    )
    origins = origins_all[selected]
    expected_origins = pd.date_range(
        f"2020-01-01 {issue_hour:02d}:00:00",
        f"2020-12-31 {issue_hour:02d}:00:00",
        freq="D",
    )
    if not origins.equals(expected_origins):
        raise ValueError("The 2020 weather archive does not contain every daily origin")

    histories: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for origin in origins:
        history_index = pd.date_range(
            origin - pd.Timedelta(hours=71), origin, freq="h"
        )
        target_index = pd.date_range(
            origin + pd.Timedelta(hours=1), periods=24, freq="h"
        )
        history = wind_frame.reindex(history_index)[provinces]
        target = wind_frame.reindex(target_index)[provinces]
        if history.isna().any().any() or target.isna().any().any():
            raise ValueError(f"Wind data is incomplete around origin {origin}")
        histories.append(history.to_numpy(dtype=np.float32))
        targets.append(target.to_numpy(dtype=np.float32))

    weather = np.asarray(archive["forecast"][selected], dtype=np.float32)
    return (
        origins.to_numpy(dtype="datetime64[ns]"),
        np.stack(histories),
        weather,
        np.stack(targets),
        provinces,
    )


def training_normalization() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover the exact training-only statistics used by ``main_train.py``."""
    train_loader, _ = make_2019_loaders(batch_size=32, seed=42)
    dataset = train_loader.dataset
    return (
        dataset.wind_mean.numpy(),
        dataset.wind_std.numpy(),
        dataset.weather_mean.numpy(),
        dataset.weather_std.numpy(),
    )


@torch.inference_mode()
def predict_model(
    model_name: str,
    checkpoint_path: Path,
    history: np.ndarray,
    weather: np.ndarray,
    wind_mean: np.ndarray,
    wind_std: np.ndarray,
    weather_mean: np.ndarray,
    weather_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Load one checkpoint and return forecasts in MW."""
    model = build_model(model_name).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    history_scaled = (history - wind_mean[None, None, :]) / wind_std[None, None, :]
    weather_scaled = (
        weather - weather_mean[None, None, None, :]
    ) / weather_std[None, None, None, :]
    predictions: list[np.ndarray] = []

    for start in range(0, len(history_scaled), batch_size):
        end = start + batch_size
        history_batch = torch.from_numpy(history_scaled[start:end]).to(device)
        weather_batch = torch.from_numpy(weather_scaled[start:end]).to(device)
        prediction_scaled = model(history_batch, weather_batch)
        prediction = (
            prediction_scaled * torch.as_tensor(wind_std, device=device)[None, None, :]
            + torch.as_tensor(wind_mean, device=device)[None, None, :]
        )
        # Wind generation cannot be negative. No upper clipping is imposed
        # because installed-capacity metadata is not part of this dataset.
        predictions.append(prediction.clamp_min(0).cpu().numpy())

    return np.concatenate(predictions, axis=0)


def forecast_frame(
    origins: np.ndarray,
    predictions: np.ndarray,
    provinces: list[str],
) -> pd.DataFrame:
    """Convert [day, lead, province] predictions to a documented table."""
    origin_index = pd.DatetimeIndex(origins)
    repeated_origins = np.repeat(origin_index.to_numpy(), 24)
    leads = np.tile(np.arange(1, 25, dtype=np.int16), len(origin_index))
    valid_times = repeated_origins + leads.astype("timedelta64[h]")
    frame = pd.DataFrame(predictions.reshape(-1, len(provinces)), columns=provinces)
    frame.insert(0, "valid_time_beijing", valid_times)
    frame.insert(0, "lead_hour", leads)
    frame.insert(0, "forecast_origin_beijing", repeated_origins)
    return frame


def plot_168_hours(
    frames: dict[str, pd.DataFrame],
    actual: np.ndarray,
    origins: np.ndarray,
    figure_dir: Path,
    start: str = "2020-07-01 00:00:00",
) -> tuple[Path, Path]:
    """Plot a continuous seven-day national forecast in a 1x4 layout."""
    start_time = pd.Timestamp(start)
    end_time = start_time + pd.Timedelta(hours=167)
    actual_times = np.concatenate(
        [
            pd.date_range(pd.Timestamp(origin) + pd.Timedelta(hours=1), periods=24, freq="h")
            .to_numpy()
            for origin in origins
        ]
    )
    actual_total = actual.reshape(-1, actual.shape[-1]).sum(axis=1) / 1000.0
    actual_series = pd.Series(actual_total, index=pd.DatetimeIndex(actual_times))
    actual_week = actual_series.loc[start_time:end_time]
    if len(actual_week) != 168:
        raise ValueError("Selected figure interval does not contain 168 hours")

    colors = {
        "cnn": "#D55E00",
        "lstm": "#D55E00",
        "transformer": "#D55E00",
        "dnn": "#D55E00",
    }
    labels = {"cnn": "CNN", "lstm": "LSTM", "transformer": "Transformer", "dnn": "DNN"}
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35), sharex=True, sharey=True)

    for name, axis in zip(MODEL_NAMES, axes):
        frame = frames[name].set_index("valid_time_beijing")
        prediction_week = frame.loc[start_time:end_time].iloc[:, 2:].sum(axis=1) / 1000.0
        if len(prediction_week) != 168:
            raise ValueError(f"{name} figure interval does not contain 168 hours")
        error = prediction_week.to_numpy() - actual_week.to_numpy()
        mae = np.mean(np.abs(error))

        axis.plot(actual_week.index, actual_week, color="#0072B2", lw=1.0, label="Observed")
        axis.plot(
            prediction_week.index,
            prediction_week,
            color=colors[name],
            lw=1.0,
            label="Forecast",
        )
        axis.text(0.5, 0.96, labels[name], transform=axis.transAxes, va="top", ha="center")
        axis.text(
            0.97,
            0.05,
            f"MAE = {mae:.2f} GW",
            transform=axis.transAxes,
            va="bottom",
            ha="right",
            fontsize=6.5,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.7)
        axis.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        axis.tick_params(axis="x", rotation=45)

    axes[0].set_ylabel("National wind power (GW)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        fontsize=7,
        handlelength=2.8,
        columnspacing=1.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=0.8)

    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / "wind_forecast_2020_168h_1x4.png"
    pdf_path = figure_dir / "wind_forecast_2020_168h_1x4.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/forecast_2020"))
    parser.add_argument("--figure-dir", type=Path, default=Path("results/diagnostics"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--issue-hour", type=int, default=8)
    parser.add_argument("--figure-start", default="2020-07-01 00:00:00")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    origins, history, weather, actual, provinces = load_2020_daily_inputs(args.issue_hour)
    wind_mean, wind_std, weather_mean, weather_std = training_normalization()
    frames: dict[str, pd.DataFrame] = {}

    print(f"device={device} daily_origins={len(origins)}")
    for model_name in MODEL_NAMES:
        checkpoint = args.checkpoint_dir / f"{model_name}_best.pt"
        prediction = predict_model(
            model_name,
            checkpoint,
            history,
            weather,
            wind_mean,
            wind_std,
            weather_mean,
            weather_std,
            device,
            args.batch_size,
        )
        frame = forecast_frame(origins, prediction, provinces)
        output_path = args.output_dir / f"{model_name}_forecast_2020.csv"
        frame.to_csv(output_path, index=False, float_format="%.3f")
        frames[model_name] = frame
        print(f"saved {model_name}: {output_path} rows={len(frame)}")

    combined = pd.concat(
        [frame.assign(model=model_name) for model_name, frame in frames.items()],
        ignore_index=True,
    )
    combined.insert(0, "model", combined.pop("model"))
    combined_path = PROJECT_ROOT / "results" / "wind_forecast_2020_all_models.csv"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_path, index=False, float_format="%.3f")
    print(f"saved combined forecasts: {combined_path} rows={len(combined)}")

    png_path, pdf_path = plot_168_hours(
        frames, actual, origins, args.figure_dir, args.figure_start
    )
    print(f"saved figure: {png_path}")
    print(f"saved figure: {pdf_path}")


if __name__ == "__main__":
    main()

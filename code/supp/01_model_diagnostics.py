"""Supplementary model-training and representative-period diagnostics.

Creates two 2 x 2 figures for CNN, LSTM, Transformer and DNN:
training/validation loss curves and national wind-power forecasts over an
objectively selected representative 30-day period.  Matplotlib defaults
are retained except for enlarged fonts and figure layout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "supp" / "supp_figure"
FORECAST = RESULTS / "wind_forecast_2020_all_models.csv"
ACTUAL = ROOT / "dataset" / "provincial_hourly_2018_2023" / "provincial_hourly_wind_2018_2023.csv"

MODELS = ("CNN", "LSTM", "Transformer", "DNN")
MODEL_KEYS = {"CNN": "cnn", "LSTM": "lstm", "Transformer": "transformer", "DNN": "dnn"}
PANEL_LABELS = ("a", "b", "c", "d")


def load_histories() -> pd.DataFrame:
    frames = []
    for model in MODELS:
        history = pd.read_csv(RESULTS / f"{MODEL_KEYS[model]}_history.csv")
        history.insert(0, "model", model)
        frames.append(history)
    return pd.concat(frames, ignore_index=True)


def build_representative_period() -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    forecast = pd.read_csv(FORECAST, parse_dates=["valid_time_beijing"])
    actual = pd.read_csv(ACTUAL, parse_dates=["timestamp_beijing"]).set_index("timestamp_beijing")
    metadata = {"model", "forecast_origin_beijing", "lead_hour", "valid_time_beijing"}
    provinces = [column for column in forecast.columns if column not in metadata]

    calendar = pd.date_range("2020-01-01", "2020-12-31 23:00", freq="h")
    calendar = calendar[~((calendar.month == 2) & (calendar.day == 29))]
    rows = []
    for model in MODELS:
        block = forecast.loc[forecast.model == MODEL_KEYS[model]].sort_values("valid_time_beijing")
        block = block.loc[~((block.valid_time_beijing.dt.month == 2) &
                            (block.valid_time_beijing.dt.day == 29))].copy()
        if len(block) != len(calendar):
            raise ValueError(f"{model} has {len(block)} forecast hours, expected {len(calendar)}")
        times = pd.DatetimeIndex(block.valid_time_beijing)
        observation = actual.reindex(times)[provinces]
        if observation.isna().any().any():
            raise ValueError(f"Observed wind data do not cover {model}")
        rows.append(pd.DataFrame({
            "model": model,
            "analysis_time": calendar,
            "observed_gw": observation.sum(axis=1).to_numpy() / 1e3,
            "forecast_gw": block[provinces].sum(axis=1).to_numpy() / 1e3,
        }))
    hourly = pd.concat(rows, ignore_index=True)
    hourly["date"] = hourly.analysis_time.dt.normalize()
    daily = (
        hourly.assign(abs_error=lambda x: (x.forecast_gw - x.observed_gw).abs())
        .groupby(["date", "model"], observed=True).abs_error.mean().unstack("model")
    )
    daily["four_model_mae_gw"] = daily[list(MODELS)].mean(axis=1)
    window_length = 30
    window_mae = daily.four_model_mae_gw.rolling(window_length).mean().dropna()
    median_mae = window_mae.median()
    end_position = daily.index.get_loc((window_mae - median_mae).abs().idxmin())
    selected_dates = daily.index[end_position - window_length + 1:end_position + 1]
    start_date = pd.Timestamp(selected_dates[0])
    end_date = pd.Timestamp(selected_dates[-1])
    selected = hourly.loc[hourly.date.isin(selected_dates)].copy()
    for model in MODELS:
        mask = selected.model == model
        selected.loc[mask, "elapsed_hour"] = np.arange(mask.sum())
    selection = daily.reset_index()
    selection["window_30d_mae_gw"] = daily.four_model_mae_gw.rolling(window_length).mean().to_numpy()
    return selected, start_date, end_date, selection


def make_loss_figure(histories: pd.DataFrame) -> plt.Figure:
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 15})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for ax, model, label in zip(axes.flat, MODELS, PANEL_LABELS):
        data = histories.loc[histories.model == model]
        ax.plot(data.epoch, data.train_mse / 1e6, label="Training")
        validation_line, = ax.plot(data.epoch, data.val_mse / 1e6, label="Validation")
        best = data.loc[data.val_mse.idxmin()]
        ax.scatter(best.epoch, best.val_mse / 1e6, color=validation_line.get_color())
        ax.text(0.97, 0.92, f"Best epoch: {int(best.epoch)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=13)
        ax.set_title(model, fontsize=19)
        ax.text(-0.10, 1.04, label, transform=ax.transAxes,
                fontsize=20, fontweight="bold")
        ax.tick_params(labelsize=13)
    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch", fontsize=17)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"MSE ($10^6$ MW$^2$)", fontsize=17)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=15)
    fig.suptitle("Training and validation loss", fontsize=21, y=0.94)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.09, top=0.84,
                        wspace=0.16, hspace=0.24)
    return fig


def make_forecast_figure(selected: pd.DataFrame, start_date: pd.Timestamp,
                         end_date: pd.Timestamp) -> plt.Figure:
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 15})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for ax, model, label in zip(axes.flat, MODELS, PANEL_LABELS):
        data = selected.loc[selected.model == model].sort_values("elapsed_hour")
        elapsed_day = data.elapsed_hour.to_numpy() / 24.0
        ax.plot(elapsed_day, data.observed_gw, label="Observed")
        ax.plot(elapsed_day, data.forecast_gw, label="Forecast")
        ax.set_title(model, fontsize=19)
        ax.set_xlim(0, 30)
        ax.set_xticks(np.arange(0, 31, 5))
        ax.text(-0.10, 1.04, label, transform=ax.transAxes,
                fontsize=20, fontweight="bold")
        ax.tick_params(labelsize=13)
    for ax in axes[-1, :]:
        ax.set_xlabel("Day", fontsize=17)
    for ax in axes[:, 0]:
        ax.set_ylabel("National wind power (GW)", fontsize=17)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=15)
    fig.suptitle(f"Representative 30-day period ({start_date:%Y-%m-%d} to {end_date:%Y-%m-%d})",
                 fontsize=21, y=0.94)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.09, top=0.84,
                        wspace=0.16, hspace=0.24)
    return fig


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.pdf")
    fig.savefig(OUTPUT / f"{stem}.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    histories = load_histories()
    selected, start_date, end_date, daily = build_representative_period()
    histories.to_csv(OUTPUT / "01_training_losses_values.csv", index=False)
    selected.to_csv(OUTPUT / "01_30day_forecasts_values.csv", index=False)
    daily.to_csv(OUTPUT / "01_30day_selection_metrics.csv", index=False)
    save_figure(make_loss_figure(histories), "01_training_losses")
    save_figure(make_forecast_figure(selected, start_date, end_date), "01_30day_forecasts")
    print(f"Representative period: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}")
    print(f"Saved {OUTPUT / '01_training_losses.pdf'}")
    print(f"Saved {OUTPUT / '01_30day_forecasts.pdf'}")


if __name__ == "__main__":
    main()

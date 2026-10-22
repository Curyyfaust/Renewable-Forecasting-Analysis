"""Supplementary Figure 02: monthly balancing outcomes for four models.

The upper block shows daily upward and downward balancing energy within each
month.  The lower block summarizes monthly balancing cost and the absolute
monthly change in direct CO2 emissions relative to the perfect-forecast run.

Plot styling intentionally uses Matplotlib defaults.  Only font sizes and the
figure layout are changed to keep the supplementary figure legible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DISPATCH = ROOT / "results" / "pypsa_dispatch_2020_all_models.csv"
PERFECT = ROOT / "results" / "pypsa_two_stage_2020" / "perfect_forecast_hourly.csv"
OUTPUT = ROOT / "supp" / "supp_figure"

MODELS = ("CNN", "LSTM", "Transformer", "DNN")
MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def load_and_aggregate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return daily balancing energy and monthly cost/emission summaries."""
    hourly = pd.read_csv(DISPATCH, parse_dates=["timestamp_beijing"])
    perfect = pd.read_csv(PERFECT, parse_dates=["timestamp_beijing"])

    required = {
        "model", "timestamp_beijing", "up_balancing_mw", "down_balancing_mw",
        "balancing_cost_cny", "direct_co2_t",
    }
    missing = required.difference(hourly.columns)
    if missing:
        raise ValueError(f"Dispatch table is missing columns: {sorted(missing)}")
    if set(hourly["model"].unique()) != set(MODELS):
        raise ValueError(f"Expected {MODELS}, found {sorted(hourly['model'].unique())}")

    baseline = perfect.set_index("timestamp_beijing")["direct_co2_t"].rename("perfect_co2_t")
    hourly = hourly.merge(baseline, left_on="timestamp_beijing", right_index=True, how="left")
    if hourly["perfect_co2_t"].isna().any():
        raise ValueError("Perfect-forecast CO2 baseline does not cover every hour")

    # The experiment has 8,760 analysed hours and omits leap day.  Assigning a
    # common analysis calendar prevents the 09:00 start time from shifting the
    # last eight hours of each month into the following calendar month.
    calendar = pd.date_range("2020-01-01", "2020-12-31 23:00", freq="h")
    calendar = calendar[~((calendar.month == 2) & (calendar.day == 29))]
    hourly = hourly.sort_values(["model", "timestamp_beijing"]).reset_index(drop=True)
    counts = hourly.groupby("model", observed=True).size()
    if not (counts == len(calendar)).all():
        raise ValueError(f"Expected {len(calendar)} hours per model, found {counts.to_dict()}")
    hourly["analysis_time"] = np.tile(calendar.to_numpy(), len(MODELS))
    hourly["delta_co2_t"] = hourly["direct_co2_t"] - hourly["perfect_co2_t"]

    daily = (
        hourly.groupby(["model", hourly.analysis_time.dt.month.rename("month"),
                         hourly.analysis_time.dt.day.rename("day")], observed=True)
        .agg(up_gwh=("up_balancing_mw", lambda x: x.sum() / 1e3),
             down_gwh=("down_balancing_mw", lambda x: x.sum() / 1e3))
        .reset_index()
    )
    monthly = (
        hourly.groupby(["model", hourly.analysis_time.dt.month.rename("month")], observed=True)
        .agg(balancing_cost_m_cny=("balancing_cost_cny", lambda x: x.sum() / 1e6),
             delta_co2_t=("delta_co2_t", "sum"))
        .reset_index()
    )
    monthly["absolute_delta_co2_mt"] = monthly["delta_co2_t"].abs() / 1e6
    return daily, monthly


def draw_daily_block(fig: plt.Figure, outer, daily: pd.DataFrame) -> list[plt.Axes]:
    """Draw 12 monthly pairs of daily up/down balancing series."""
    grid = outer.subgridspec(6, 4, hspace=0.22, wspace=0.16)
    axes: list[plt.Axes] = []
    for month in range(1, 13):
        row = (month - 1) // 4
        col = (month - 1) % 4
        ax_up = fig.add_subplot(grid[2 * row, col])
        ax_down = fig.add_subplot(grid[2 * row + 1, col], sharex=ax_up)
        axes.extend([ax_up, ax_down])

        for model in MODELS:
            values = daily.loc[(daily.model == model) & (daily.month == month)]
            ax_up.plot(values.day, values.up_gwh, label=model)
            ax_down.plot(values.day, values.down_gwh, label=model)

        ax_up.set_title(MONTH_NAMES[month - 1], fontsize=18)
        ax_up.text(0.97, 0.86, "Up", ha="right", va="top",
                   transform=ax_up.transAxes, fontsize=15)
        ax_down.text(0.97, 0.86, "Down", ha="right", va="top",
                     transform=ax_down.transAxes, fontsize=15)
        ax_up.tick_params(labelbottom=False)
        ax_up.set_xlim(0.5, 31.5)
        ax_down.set_xticks([1, 15, 30])
        if row < 2:
            ax_down.tick_params(labelbottom=False)
        if col != 0:
            ax_up.tick_params(labelleft=False)
            ax_down.tick_params(labelleft=False)

    # Figure-level labels preserve uncluttered small multiples.
    fig.text(0.025, 0.54, "Daily balancing (GWh)", rotation=90,
             ha="center", va="center", fontsize=18)
    fig.text(0.5, 0.055, "Day of month", ha="center", va="center", fontsize=18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.035),
               ncol=4, fontsize=16)
    return axes


def draw_monthly_block(fig: plt.Figure, outer, monthly: pd.DataFrame) -> list[plt.Axes]:
    """Draw grouped monthly bars for cost and absolute CO2 change."""
    grid = outer.subgridspec(1, 2, wspace=0.22)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    x = np.arange(1, 13, dtype=float)
    width = 0.18
    offsets = (np.arange(len(MODELS)) - 1.5) * width

    for index, model in enumerate(MODELS):
        values = monthly.loc[monthly.model == model].sort_values("month")
        axes[0].bar(x + offsets[index], values.balancing_cost_m_cny, width, label=model)
        axes[1].bar(x + offsets[index], values.absolute_delta_co2_mt, width, label=model)

    for ax in axes:
        ax.set_xlim(0.4, 12.6)
        ax.set_xticks([1, 4, 7, 10], ["Jan", "Apr", "Jul", "Oct"])
        ax.set_xlabel("Month", fontsize=17)
    axes[0].set_ylabel("Monthly balancing cost (M CNY)", fontsize=17)
    axes[1].set_ylabel(r"Monthly $|\Delta \mathrm{CO}_2|$ (Mt)", fontsize=17)
    axes[0].text(-0.13, 1.04, "a", transform=axes[0].transAxes,
                 fontsize=20, fontweight="bold")
    axes[1].text(-0.13, 1.04, "b", transform=axes[1].transAxes,
                 fontsize=20, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97),
               ncol=4, fontsize=16)
    return axes


def make_daily_figure(daily: pd.DataFrame) -> plt.Figure:
    """Build the daily balancing figure using default Matplotlib style."""
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 14})
    fig = plt.figure(figsize=(18, 15))
    outer = fig.add_gridspec(1, 1, left=0.07, right=0.98, top=0.97, bottom=0.10)
    draw_daily_block(fig, outer[0], daily)
    return fig


def make_monthly_figure(monthly: pd.DataFrame) -> plt.Figure:
    """Build the monthly cost and CO2 figure using default Matplotlib style."""
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 14})
    fig = plt.figure(figsize=(16, 6))
    outer = fig.add_gridspec(1, 1, left=0.08, right=0.98, top=0.82, bottom=0.16)
    draw_monthly_block(fig, outer[0], monthly)
    return fig


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    daily, monthly = load_and_aggregate()
    daily.to_csv(OUTPUT / "02_supp_daily_values.csv", index=False)
    monthly.to_csv(OUTPUT / "02_supp_monthly_values.csv", index=False)
    daily_fig = make_daily_figure(daily)
    daily_fig.savefig(OUTPUT / "02_daily_balancing.pdf")
    daily_fig.savefig(OUTPUT / "02_daily_balancing.png", dpi=220)
    plt.close(daily_fig)

    monthly_fig = make_monthly_figure(monthly)
    monthly_fig.savefig(OUTPUT / "02_monthly_impacts.pdf")
    monthly_fig.savefig(OUTPUT / "02_monthly_impacts.png", dpi=220)
    plt.close(monthly_fig)
    print(f"Saved {OUTPUT / '02_daily_balancing.pdf'}")
    print(f"Saved {OUTPUT / '02_daily_balancing.png'}")
    print(f"Saved {OUTPUT / '02_monthly_impacts.pdf'}")
    print(f"Saved {OUTPUT / '02_monthly_impacts.png'}")


if __name__ == "__main__":
    main()

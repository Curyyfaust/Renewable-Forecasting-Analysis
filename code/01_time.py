"""Time-scale wind-forecast and dispatch diagnostics for the 2020 experiment.

The script reproduces the five-panel layout supplied as a visual reference and
writes both the figure and the values used to draw it.  All cost differences
and emission differences are measured against the perfect-forecast dispatch.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DISPATCH = ROOT / "results" / "pypsa_dispatch_2020_all_models.csv"
PERFECT = ROOT / "results" / "pypsa_two_stage_2020" / "perfect_forecast_hourly.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figure"

MODELS = ("CNN", "LSTM", "Transformer", "DNN")
COLORS = {
    "CNN": "#1787BE",
    "LSTM": "#F29B00",
    "Transformer": "#9567CC",
    "DNN": "#00A37A",
}
SHORT = {"CNN": "CNN", "LSTM": "LSTM", "Transformer": "Transformer", "DNN": "DNN"}
SEASONS = ("Winter", "Spring", "Summer", "Autumn")
SEASON_COLORS = {
    "Winter": "#4C78A8",
    "Spring": "#54A24B",
    "Summer": "#EFC34A",
    "Autumn": "#B279A2",
}
AGGREGATIONS = {"Hourly": "h", "Daily": "D", "Weekly": "W", "Monthly": "ME"}


def load_data() -> pd.DataFrame:
    """Load four model runs and attach the perfect-forecast baseline."""
    hourly = pd.read_csv(DISPATCH, parse_dates=["timestamp_beijing"])
    perfect = pd.read_csv(PERFECT, parse_dates=["timestamp_beijing"])

    required = {
        "model", "timestamp_beijing", "forecast_wind_mw", "observed_wind_mw",
        "forecast_error_mw", "up_balancing_mw", "down_balancing_mw",
        "balancing_cost_cny", "realised_fuel_cost_cny", "direct_co2_t",
        "total_operating_cost_cny",
    }
    missing = required.difference(hourly.columns)
    if missing:
        raise ValueError(f"Dispatch input is missing columns: {sorted(missing)}")
    if set(hourly["model"].unique()) != set(MODELS):
        raise ValueError(f"Expected models {MODELS}, found {sorted(hourly['model'].unique())}")

    baseline = perfect.set_index("timestamp_beijing")[[
        "realised_fuel_cost_cny", "direct_co2_t", "total_operating_cost_cny"
    ]].rename(columns={
        "realised_fuel_cost_cny": "perfect_fuel_cost_cny",
        "direct_co2_t": "perfect_co2_t",
        "total_operating_cost_cny": "perfect_operating_cost_cny",
    })
    hourly = hourly.merge(baseline, left_on="timestamp_beijing", right_index=True, how="left")
    if hourly[["perfect_fuel_cost_cny", "perfect_co2_t"]].isna().any().any():
        raise ValueError("Perfect-forecast baseline does not cover every dispatch hour")

    hourly["delta_fuel_cost_cny"] = (
        hourly["realised_fuel_cost_cny"] - hourly["perfect_fuel_cost_cny"]
    )
    hourly["delta_co2_t"] = hourly["direct_co2_t"] - hourly["perfect_co2_t"]
    if "total_cost_difference_cny" not in hourly:
        hourly["total_cost_difference_cny"] = (
            hourly["total_operating_cost_cny"] - hourly["perfect_operating_cost_cny"]
        )

    # The experiment contains 8,760 consecutive analysed hours but its valid
    # times run from 09:00 to 08:00 and omit leap day.  This calendar gives all
    # models the same clean 2020 month/season labels without changing row order.
    count = hourly.groupby("model").size()
    if not (count == 8760).all():
        raise ValueError(f"Each model must contain 8760 hours, found {count.to_dict()}")
    analysis_calendar = pd.date_range("2020-01-01", "2020-12-31 23:00", freq="h")
    analysis_calendar = analysis_calendar[~((analysis_calendar.month == 2) & (analysis_calendar.day == 29))]
    hourly = hourly.sort_values(["model", "timestamp_beijing"]).reset_index(drop=True)
    hourly["analysis_time"] = np.tile(analysis_calendar.to_numpy(), len(MODELS))
    hourly["season"] = pd.Categorical(
        np.select(
            [hourly.analysis_time.dt.month.isin([12, 1, 2]),
             hourly.analysis_time.dt.month.isin([3, 4, 5]),
             hourly.analysis_time.dt.month.isin([6, 7, 8])],
            ["Winter", "Spring", "Summer"], default="Autumn",
        ), categories=SEASONS, ordered=True,
    )
    return hourly


def resample_model(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Aggregate power to period means and energy/cost/emissions to sums."""
    x = frame.set_index("analysis_time").sort_index()
    mean_cols = ["forecast_wind_mw", "observed_wind_mw", "forecast_error_mw"]
    sum_cols = [
        "up_balancing_mw", "down_balancing_mw", "balancing_cost_cny",
        "delta_fuel_cost_cny", "delta_co2_t", "total_cost_difference_cny",
    ]
    if frequency == "h":
        result = x[mean_cols + sum_cols].copy()
        result["hours"] = 1
        return result
    means = x[mean_cols].resample(frequency).mean()
    sums = x[sum_cols].resample(frequency).sum()
    hours = x["forecast_error_mw"].resample(frequency).count().rename("hours")
    return means.join(sums).join(hours).dropna(subset=mean_cols)


def r_squared(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1] ** 2)


def slope_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    denominator = float(np.dot(x[valid], x[valid]))
    return float(np.dot(x[valid], y[valid]) / denominator) if denominator else np.nan


def error_metrics(frame: pd.DataFrame, normalizer: float) -> dict[str, float]:
    error = frame["forecast_error_mw"].to_numpy(float)
    hours = frame["hours"].to_numpy(float) if "hours" in frame else np.ones(len(frame))
    cost = frame["total_cost_difference_cny"].to_numpy(float) / hours / 1e6
    error_gw = np.abs(error) / 1e3
    return {
        "nmae_pct": float(np.mean(np.abs(error)) / normalizer * 100),
        "nrmse_pct": float(np.sqrt(np.mean(error ** 2)) / normalizer * 100),
        "abs_bias_pct": float(abs(np.mean(error)) / normalizer * 100),
        "error_cost_r2": r_squared(error_gw, cost),
        "cost_sensitivity_m_cny_per_gw": slope_through_origin(error_gw, cost),
    }


def build_metrics(hourly: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """Calculate panel c/d metrics and retain distributions for plotting."""
    rows: list[dict[str, float | str | int]] = []
    distributions: dict[tuple[str, str], pd.DataFrame] = {}
    observed_mean = float(hourly.loc[hourly.model == MODELS[0], "observed_wind_mw"].mean())

    for model in MODELS:
        model_hourly = hourly.loc[hourly.model == model].copy()
        for label, frequency in AGGREGATIONS.items():
            aggregated = resample_model(model_hourly, frequency)
            distributions[(model, label)] = aggregated
            rows.append({"model": model, "condition": label, "samples": len(aggregated),
                         **error_metrics(aggregated, observed_mean)})

        ramp = model_hourly["observed_wind_mw"].diff().abs()
        high_ramp = model_hourly.loc[ramp >= ramp.quantile(0.90)].set_index("analysis_time")
        low_wind = model_hourly.loc[
            model_hourly.observed_wind_mw <= model_hourly.observed_wind_mw.quantile(0.10)
        ].set_index("analysis_time")
        for label, subset in (("High ramp", high_ramp), ("Low wind", low_wind)):
            distributions[(model, label)] = subset
            rows.append({"model": model, "condition": label, "samples": len(subset),
                         **error_metrics(subset, observed_mean)})
    return pd.DataFrame(rows), distributions


def style_axis(ax: plt.Axes, grid: bool = False) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.05)
        spine.set_color("#333333")
    ax.tick_params(width=1.0, length=4, color="#333333")
    if grid:
        ax.grid(axis="y", color="#C8CED3", linestyle=":", linewidth=0.65, alpha=0.75)
        ax.set_axisbelow(True)


def colored_boxplot(ax: plt.Axes, values: list[np.ndarray], positions: np.ndarray,
                    colors: list[str], width: float = 0.15) -> None:
    bp = ax.boxplot(values, positions=positions, widths=width, patch_artist=True,
                    showfliers=False, manage_ticks=False,
                    medianprops={"linewidth": 1.4}, whiskerprops={"linewidth": 1.0},
                    capprops={"linewidth": 1.0})
    for box, color in zip(bp["boxes"], colors):
        box.set(facecolor="white", edgecolor=color, linewidth=1.6)
    for median, color in zip(bp["medians"], colors):
        median.set(color=color)
    for index, color in enumerate(colors):
        for line in (bp["whiskers"][2 * index:2 * index + 2] + bp["caps"][2 * index:2 * index + 2]):
            line.set(color=color)


def panel_a(fig: plt.Figure, spec, hourly: pd.DataFrame) -> None:
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=spec, hspace=0.08, height_ratios=[1.25, 1])
    top = fig.add_subplot(inner[0])
    bottom = fig.add_subplot(inner[1], sharex=top)
    twin = top.twinx()
    daily_frames = []
    for model in MODELS:
        model_frame = hourly.loc[hourly.model == model].set_index("analysis_time")
        values = model_frame.resample("D").agg(
            delta_fuel_cost_cny=("delta_fuel_cost_cny", "sum"),
            balancing_cost_cny=("balancing_cost_cny", "sum"),
            delta_co2_t=("delta_co2_t", "sum"),
            season=("season", "first"),
        ).reset_index()
        values.insert(0, "model", model)
        daily_frames.append(values)
    daily = pd.concat(daily_frames, ignore_index=True)
    x = np.arange(len(MODELS), dtype=float)
    fuel = [daily.loc[daily.model == m, "delta_fuel_cost_cny"].to_numpy() / 1e6 for m in MODELS]
    balance = [daily.loc[daily.model == m, "balancing_cost_cny"].to_numpy() / 1e6 for m in MODELS]
    co2 = [daily.loc[daily.model == m, "delta_co2_t"].to_numpy() / 1e3 for m in MODELS]
    colored_boxplot(top, fuel, x - 0.13, ["#80B8D4"] * 4, width=0.23)
    colored_boxplot(twin, balance, x + 0.13, ["#F28E3B"] * 4, width=0.23)
    colored_boxplot(bottom, co2, x, ["#59B89F"] * 4, width=0.36)

    offsets = np.linspace(-0.075, 0.075, 4)
    for i, model in enumerate(MODELS):
        subset = daily.loc[daily.model == model]
        for offset, season in zip(offsets, SEASONS):
            part = subset.loc[subset.season == season]
            top.scatter(i - 0.13 + offset, part.delta_fuel_cost_cny.mean() / 1e6,
                        marker="D", s=34, color=SEASON_COLORS[season], edgecolor="white", linewidth=.55, zorder=5)
            twin.scatter(i + 0.13 + offset, part.balancing_cost_cny.mean() / 1e6,
                         marker="D", s=34, color=SEASON_COLORS[season], edgecolor="white", linewidth=.55, zorder=5)
            bottom.scatter(i + offset, part.delta_co2_t.mean() / 1e3,
                           marker="D", s=34, color=SEASON_COLORS[season], edgecolor="white", linewidth=.55, zorder=5)

    top.axhline(0, color="#777777", lw=.75, ls=":")
    bottom.axhline(0, color="#777777", lw=.75, ls=":")
    top.set_ylabel(r"$\Delta$Fuel cost (M CNY d$^{-1}$)")
    twin.set_ylabel(r"Balancing cost (M CNY d$^{-1}$)")
    bottom.set_ylabel(r"$\Delta$CO$_2$ (kt d$^{-1}$)")
    bottom.set_xticks(x, [SHORT[m] for m in MODELS], rotation=25, ha="right")
    top.tick_params(labelbottom=False)
    top.legend(handles=[Patch(facecolor="#80B8D4", label=r"$\Delta$Fuel cost"),
                        Patch(facecolor="#F28E3B", label="Balancing cost")],
               loc="upper left", frameon=False, ncol=2, handlelength=.9,
               handletextpad=.35, columnspacing=.8, fontsize=13.5)
    bottom.legend(handles=[Line2D([0], [0], marker="D", color="none",
                                  markerfacecolor=SEASON_COLORS[s], markeredgecolor="white",
                                  label=s, markersize=5.5) for s in SEASONS],
                  loc="upper left", frameon=False, ncol=4, handlelength=.5,
                  handletextpad=.25, columnspacing=.55, fontsize=12)
    for ax in (top, twin, bottom):
        style_axis(ax)
    bottom.tick_params(axis="x", labelsize=14)
    top.text(-0.08, 1.10, "a", transform=top.transAxes, fontsize=16, fontweight="bold")


def panel_b(fig: plt.Figure, spec, hourly: pd.DataFrame) -> None:
    grid = GridSpecFromSubplotSpec(2, 4, subplot_spec=spec, hspace=.17, wspace=.18)
    month_colors = ["#DCECF5", "#EAF1DA", "#FFF1C9", "#EEE5F2"]
    months = np.arange(1, 13)
    for column, model in enumerate(MODELS):
        x = hourly.loc[hourly.model == model].set_index("analysis_time")
        monthly = x[["up_balancing_mw", "down_balancing_mw"]].resample("ME").sum() / 1e6
        for row, (field, label) in enumerate((("up_balancing_mw", "Up"), ("down_balancing_mw", "Down"))):
            ax = fig.add_subplot(grid[row, column])
            for start, end, color in ((.5, 2.5, month_colors[0]), (2.5, 5.5, month_colors[1]),
                                      (5.5, 8.5, month_colors[2]), (8.5, 12.5, month_colors[3])):
                ax.axvspan(start, end, color=color, zorder=0)
            values = monthly[field].to_numpy()
            ax.plot(months, values, color=COLORS[model], lw=2.0)
            ax.text(.04, .96, f"{values.max():.2f}", transform=ax.transAxes, va="top", fontsize=10.5)
            ax.text(.04, .04, f"{values.min():.2f}", transform=ax.transAxes, va="bottom", fontsize=10.5)
            ax.set_xlim(.5, 12.5)
            pad = max(np.ptp(values) * .12, values.max() * .05, .001)
            ax.set_ylim(values.min() - pad, values.max() + pad)
            ax.set_xticks([1, 4, 7, 10], ["J", "A", "J", "O"])
            ax.tick_params(labelleft=False, labelbottom=row == 1)
            if row == 1:
                ax.tick_params(axis="x", labelsize=13.5)
            if row == 0:
                ax.set_title(SHORT[model], fontsize=13.5, fontweight="bold", pad=2)
            if column == 0:
                ax.set_ylabel(label, fontweight="bold")
            style_axis(ax)
            if row == 0 and column == 0:
                ax.text(-.24, 1.14, "b", transform=ax.transAxes, fontsize=16, fontweight="bold")


def panel_c(fig: plt.Figure, spec, metrics: pd.DataFrame,
            distributions: dict[tuple[str, str], pd.DataFrame], normalizer: float) -> None:
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=spec, hspace=.08, height_ratios=[.63, 1.22])
    upper = fig.add_subplot(inner[0])
    lower = fig.add_subplot(inner[1], sharex=upper)
    labels = list(AGGREGATIONS) + ["High ramp", "Low wind"]
    xpos = np.arange(len(labels), dtype=float)
    upper.axvspan(-.55, 3.5, color="#EFF4F8")
    upper.axvspan(3.5, 5.55, color="#FFF7EB")
    lower.axvspan(-.55, 3.5, color="#EFF4F8")
    lower.axvspan(3.5, 5.55, color="#FFF7EB")
    for model in MODELS:
        values = [metrics.loc[(metrics.model == model) & (metrics.condition == label),
                              "error_cost_r2"].iat[0] for label in labels]
        upper.plot(xpos, values, "o-", color=COLORS[model], lw=1.05, ms=4.2, label=SHORT[model])
    upper.axvline(3.5, color="#777777", lw=.8)
    upper.set_ylim(0, 1.03)
    upper.set_ylabel("Error–cost coupling\n$R^2$")
    upper.tick_params(labelbottom=False)
    upper.text(.24, 1.08, "TEMPORAL AGGREGATION", transform=upper.transAxes,
               ha="center", fontweight="bold", fontsize=12.5)
    upper.text(.81, 1.08, "EXTREME CONDITIONS", transform=upper.transAxes,
               ha="center", fontweight="bold", fontsize=12.5)
    upper.legend(loc="upper center", bbox_to_anchor=(.49, 1.34), frameon=False,
                 ncol=4, handlelength=.6, columnspacing=1.0, handletextpad=.35, fontsize=13)
    style_axis(upper, grid=True)

    width = .15
    offsets = np.linspace(-1.5 * width, 1.5 * width, 4)
    for model_index, model in enumerate(MODELS):
        arrays = []
        for label in labels:
            frame = distributions[(model, label)]
            arrays.append((frame.forecast_error_mw.abs() / normalizer * 100).to_numpy())
        colored_boxplot(lower, arrays, xpos + offsets[model_index], [COLORS[model]] * len(labels), width=.14)
    lower.axvline(3.5, color="#777777", lw=.8)
    lower.axhline(0, color="#A0A0A0", lw=.6, ls=":")
    lower.set_ylabel("Wind forecast error\nNMAE (%)")
    lower.set_xticks(xpos, labels, rotation=17, ha="right")
    lower.tick_params(axis="x", labelsize=14)
    style_axis(lower, grid=True)
    upper.text(-.03, 1.16, "c", transform=upper.transAxes, fontsize=16, fontweight="bold")


def draw_heatmap(ax: plt.Axes, values: np.ndarray, rows: list[str], title: str,
                 cmap: str, fmt: str) -> None:
    image = ax.imshow(values, cmap=cmap, aspect="auto")
    threshold = (np.nanmin(values) + np.nanmax(values)) / 2
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] > threshold and cmap == "Reds" else "#222222"
            ax.text(j, i, format(values[i, j], fmt), ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color=color)
    ax.set_xticks(range(4), ["H", "D", "W", "M"])
    ax.set_yticks(range(4), rows)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=3)
    ax.tick_params(length=0)
    ax.tick_params(axis="x", labelsize=13.5)
    style_axis(ax)
    image.set_clim(np.nanmin(values), np.nanmax(values))


def panel_d(fig: plt.Figure, spec, metrics: pd.DataFrame) -> None:
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=spec, hspace=.55, height_ratios=[1, 1.1])
    bar = fig.add_subplot(inner[0])
    labels = list(AGGREGATIONS)
    fields = (("nmae_pct", "NMAE", "#D43D70"),
              ("nrmse_pct", "NRMSE", "#F27658"),
              ("abs_bias_pct", "|Bias|", "#F2B33F"))
    xpos = np.arange(4)
    width = .21
    for index, (field, label, color) in enumerate(fields):
        means, errors = [], []
        for condition in labels:
            values = metrics.loc[metrics.condition == condition, field].to_numpy()
            means.append(values.mean())
            errors.append(values.std(ddof=1))
        bar.bar(xpos + (index - 1) * width, means, width, yerr=errors, capsize=2,
                color=color, edgecolor="none", label=label)
    bar.set_xticks(xpos, labels)
    bar.tick_params(axis="x", labelsize=13.5)
    bar.set_ylabel("Error (%)")
    bar.legend(frameon=False, ncol=3, loc="upper right", handlelength=.8,
               handletextpad=.25, columnspacing=.65, fontsize=12.5)
    style_axis(bar)
    bar.text(-.08, 1.11, "d", transform=bar.transAxes, fontsize=16, fontweight="bold")

    heat = GridSpecFromSubplotSpec(1, 2, subplot_spec=inner[1], wspace=.23)
    nmae_ax = fig.add_subplot(heat[0])
    slope_ax = fig.add_subplot(heat[1])
    nmae = np.array([[metrics.loc[(metrics.model == model) & (metrics.condition == label),
                                  "nmae_pct"].iat[0] for label in labels] for model in MODELS])
    slope = np.array([[metrics.loc[(metrics.model == model) & (metrics.condition == label),
                                  "cost_sensitivity_m_cny_per_gw"].iat[0]
                       for label in labels] for model in MODELS])
    draw_heatmap(nmae_ax, nmae, [SHORT[m] for m in MODELS], "Wind NMAE (%)", "Blues", ".1f")
    draw_heatmap(slope_ax, slope, ["" for _ in MODELS], "Cost sensitivity", "Reds", ".3f")
    slope_ax.set_yticks([])


def panel_e(fig: plt.Figure, spec, hourly: pd.DataFrame) -> None:
    grid = GridSpecFromSubplotSpec(2, 2, subplot_spec=spec, hspace=.25, wspace=.15)
    edges = np.linspace(0, 100, 21)
    centers = (edges[:-1] + edges[1:]) / 2
    for index, model in enumerate(MODELS):
        ax = fig.add_subplot(grid[index // 2, index % 2])
        x = hourly.loc[hourly.model == model].copy()
        error = x.forecast_error_mw.abs()
        percentile = error.rank(method="average", pct=True).to_numpy() * 100
        cost = x.balancing_cost_cny.to_numpy() / 1e6
        means, lower, upper = [], [], []
        for left, right in zip(edges[:-1], edges[1:]):
            chosen = cost[(percentile > left) & (percentile <= right)]
            mean = float(np.mean(chosen))
            half = 1.96 * float(np.std(chosen, ddof=1)) / np.sqrt(len(chosen))
            means.append(mean); lower.append(mean - half); upper.append(mean + half)
        ax.fill_between(centers, lower, upper, color=COLORS[model], alpha=.16, linewidth=0)
        ax.plot(centers, means, color=COLORS[model], lw=2.0)
        ax.axvline(90, color="#666666", lw=.8, ls="--")
        ax.text(.04, .94, SHORT[model], transform=ax.transAxes, va="top",
                fontweight="bold", fontsize=12.5)
        ax.set_xlim(0, 100)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelleft=index % 2 == 0, labelbottom=index // 2 == 1)
        if index // 2 == 1:
            ax.tick_params(axis="x", labelsize=13.5)
        style_axis(ax)
        if index == 0:
            ax.text(-.22, 1.15, "e", transform=ax.transAxes, fontsize=16, fontweight="bold")
    bounds = spec.get_position(fig)
    fig.text((bounds.x0 + bounds.x1) / 2, bounds.y0 - .038,
             "Hourly wind-error percentile", ha="center", fontsize=14.5)
    fig.text(bounds.x0 - .034, (bounds.y0 + bounds.y1) / 2,
             r"Mean balancing cost (M CNY h$^{-1}$)", rotation=90,
             va="center", ha="center", fontsize=13)


def make_figure(hourly: pd.DataFrame, metrics: pd.DataFrame,
                distributions: dict[tuple[str, str], pd.DataFrame]) -> plt.Figure:
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 12.2, "axes.labelsize": 14,
        "axes.titlesize": 13.5, "xtick.labelsize": 13.5, "ytick.labelsize": 11.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(14.5, 12.5), facecolor="white")
    outer = GridSpec(3, 2, figure=fig, height_ratios=[1.05, 1.12, 1.02],
                     width_ratios=[.86, 1.24], hspace=.30, wspace=.30,
                     left=.066, right=.988, top=.978, bottom=.068)
    panel_a(fig, outer[0, 0], hourly)
    panel_b(fig, outer[0, 1], hourly)
    normalizer = float(hourly.loc[hourly.model == MODELS[0], "observed_wind_mw"].mean())
    panel_c(fig, outer[1, :], metrics, distributions, normalizer)
    panel_d(fig, outer[2, 0], metrics)
    panel_e(fig, outer[2, 1], hourly)
    return fig


def main() -> None:
    hourly = load_data()
    metrics, distributions = build_metrics(hourly)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS / "01_time_metrics.csv"
    png_path = FIGURES / "01_time.png"
    pdf_path = FIGURES / "01_time.pdf"
    metrics.to_csv(metrics_path, index=False, float_format="%.6f")
    figure = make_figure(hourly, metrics, distributions)
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(metrics.to_string(index=False))
    print(metrics_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()

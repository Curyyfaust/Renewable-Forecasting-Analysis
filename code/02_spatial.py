"""Spatial diagnostics for provincial wind-forecast dispatch impacts.

The PyPSA result table reports national hourly costs and emissions, while the
forecast table retains all 31 provinces.  For every model-hour, this script
allocates the national cost, balancing cost, and direct-CO2 differences to
provinces in proportion to their absolute wind-forecast error.  Province
results are then aggregated to seven geographic regions.  This allocation is
an attribution diagnostic; it is not a province-level re-solution of PyPSA.
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
FORECAST = ROOT / "results" / "wind_forecast_2020_all_models.csv"
ACTUAL = ROOT / "dataset" / "provincial_hourly_2018_2023" / "provincial_hourly_wind_2018_2023.csv"
PERFECT = ROOT / "results" / "pypsa_two_stage_2020" / "perfect_forecast_hourly.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figure"

MODELS = ("CNN", "LSTM", "Transformer", "DNN")
MODEL_KEYS = {"CNN": "cnn", "LSTM": "lstm", "Transformer": "transformer", "DNN": "dnn"}
MODEL_COLORS = {"CNN": "#2CA9C2", "LSTM": "#F28E42", "Transformer": "#A865E8", "DNN": "#00A37A"}
REGIONS = ("N", "E", "NW", "NE", "SW", "C", "S")
REGION_COLORS = {
    "N": "#2B83BA", "E": "#35A6A1", "NW": "#6B4C9A", "NE": "#E76F51",
    "SW": "#E9C46A", "C": "#7FA77A", "S": "#F4A261",
}
SEASONS = ("Winter", "Spring", "Summer", "Autumn")
SEASON_SHORT = {"Winter": "W", "Spring": "Sp", "Summer": "Su", "Autumn": "A"}
SEASON_COLORS = {"Winter": "#D9EAF4", "Spring": "#DCD4F4", "Summer": "#F6D6CC", "Autumn": "#BFE1D8"}
PROVINCE_REGION = {
    "Beijing": "N", "Tianjin": "N", "Hebei": "N", "Shanxi": "N", "InnerMongolia": "N",
    "Shanghai": "E", "Jiangsu": "E", "Zhejiang": "E", "Anhui": "E", "Fujian": "E",
    "Jiangxi": "E", "Shandong": "E",
    "Shaanxi": "NW", "Gansu": "NW", "Qinghai": "NW", "Ningxia": "NW", "Xinjiang": "NW",
    "Liaoning": "NE", "Jilin": "NE", "Heilongjiang": "NE",
    "Chongqing": "SW", "Sichuan": "SW", "Guizhou": "SW", "Yunnan": "SW", "Tibet": "SW",
    "Henan": "C", "Hubei": "C", "Hunan": "C",
    "Guangdong": "S", "Guangxi": "S", "Hainan": "S",
}
PROVINCE_ABBR = {
    "Anhui": "AH", "Beijing": "BJ", "Chongqing": "CQ", "Fujian": "FJ",
    "Gansu": "GS", "Guangdong": "GD", "Guangxi": "GX", "Guizhou": "GZ",
    "Hainan": "HI", "Hebei": "HE", "Heilongjiang": "HL", "Henan": "HA",
    "Hubei": "HB", "Hunan": "HN", "InnerMongolia": "NM", "Jiangsu": "JS",
    "Jiangxi": "JX", "Jilin": "JL", "Liaoning": "LN", "Ningxia": "NX",
    "Qinghai": "QH", "Shaanxi": "SN", "Shandong": "SD", "Shanghai": "SH",
    "Shanxi": "SX", "Sichuan": "SC", "Tianjin": "TJ", "Tibet": "XZ",
    "Xinjiang": "XJ", "Yunnan": "YN", "Zhejiang": "ZJ",
}


def season_labels(times: pd.DatetimeIndex) -> np.ndarray:
    """Assign seasons from the actual Beijing-time dispatch timestamps."""
    months = times.month.to_numpy()
    return np.select(
        [np.isin(months, [12, 1, 2]), np.isin(months, [3, 4, 5]), np.isin(months, [6, 7, 8])],
        ["Winter", "Spring", "Summer"], default="Autumn",
    )


def calculate_spatial_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return province-annual and region-season attribution tables."""
    dispatch_path = DISPATCH if DISPATCH.exists() else RESULTS / DISPATCH.name
    dispatch = pd.read_csv(dispatch_path, parse_dates=["timestamp_beijing"])
    forecast = pd.read_csv(FORECAST, parse_dates=["valid_time_beijing"])
    actual = pd.read_csv(ACTUAL, parse_dates=["timestamp_beijing"]).set_index("timestamp_beijing")
    perfect = pd.read_csv(PERFECT, parse_dates=["timestamp_beijing"]).set_index("timestamp_beijing")
    provinces = [column for column in forecast.columns if column not in {
        "model", "forecast_origin_beijing", "lead_hour", "valid_time_beijing"
    }]
    if set(provinces) != set(PROVINCE_REGION):
        missing = sorted(set(provinces).symmetric_difference(PROVINCE_REGION))
        raise ValueError(f"Province-region mapping mismatch: {missing}")

    annual_rows: list[dict[str, float | str]] = []
    seasonal_rows: list[dict[str, float | str]] = []
    for model in MODELS:
        national = dispatch.loc[dispatch.model == model].sort_values("timestamp_beijing").reset_index(drop=True)
        if len(national) != 8760:
            raise ValueError(f"{model} has {len(national)} dispatch hours, expected 8760")
        times = pd.DatetimeIndex(national.timestamp_beijing)
        seasons = season_labels(times)
        prediction = (
            forecast.loc[forecast.model == MODEL_KEYS[model]]
            .set_index("valid_time_beijing").reindex(times)[provinces]
        )
        observation = actual.reindex(times)[provinces]
        if prediction.isna().any().any() or observation.isna().any().any():
            raise ValueError(f"Provincial forecast/actual coverage is incomplete for {model}")
        absolute_error = np.abs(prediction.to_numpy(float) - observation.to_numpy(float))
        denominator = absolute_error.sum(axis=1, keepdims=True)
        weights = np.divide(absolute_error, denominator, out=np.zeros_like(absolute_error),
                            where=denominator > 0)
        nonzero = denominator[:, 0] > 0
        if not np.allclose(weights[nonzero].sum(axis=1), 1.0, atol=1e-10):
            raise ValueError(f"Provincial attribution weights do not sum to one for {model}")

        perfect_model = perfect.reindex(times)
        co2_difference = national.direct_co2_t.to_numpy() - perfect_model.direct_co2_t.to_numpy()
        cost_difference = national.total_cost_difference_cny.to_numpy()
        balancing_cost = national.balancing_cost_cny.to_numpy()
        allocated_cost = weights * cost_difference[:, None]
        allocated_balance = weights * balancing_cost[:, None]
        allocated_co2 = weights * co2_difference[:, None]

        for province_index, province in enumerate(provinces):
            annual_rows.append({
                "model": model,
                "province": province,
                "region": PROVINCE_REGION[province],
                "absolute_error_gwh": absolute_error[:, province_index].sum() / 1e3,
                "cost_penalty_million_cny": allocated_cost[:, province_index].sum() / 1e6,
                "balancing_cost_million_cny": allocated_balance[:, province_index].sum() / 1e6,
                "direct_co2_penalty_mt": allocated_co2[:, province_index].sum() / 1e6,
            })

        region_array = np.array([PROVINCE_REGION[p] for p in provinces])
        for season in SEASONS:
            season_mask = seasons == season
            for region in REGIONS:
                province_mask = region_array == region
                error_gwh = absolute_error[np.ix_(season_mask, province_mask)].sum() / 1e3
                cost_m = allocated_cost[np.ix_(season_mask, province_mask)].sum() / 1e6
                seasonal_rows.append({
                    "model": model,
                    "season": season,
                    "region": region,
                    "absolute_error_gwh": error_gwh,
                    "cost_penalty_million_cny": cost_m,
                    "cost_intensity_m_cny_per_gwh": (
                        allocated_balance[np.ix_(season_mask, province_mask)].sum() / 1e6 / error_gwh
                        if error_gwh else np.nan
                    ),
                    "balancing_cost_million_cny": allocated_balance[np.ix_(season_mask, province_mask)].sum() / 1e6,
                    "direct_co2_penalty_mt": allocated_co2[np.ix_(season_mask, province_mask)].sum() / 1e6,
                })
    return pd.DataFrame(annual_rows), pd.DataFrame(seasonal_rows)


def style_axis(ax: plt.Axes, grid: bool = False) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.05)
        spine.set_color("#333333")
    ax.tick_params(width=1.0, length=4, color="#333333")
    if grid:
        ax.grid(axis="y", color="#C8CED3", linestyle=":", linewidth=.65, alpha=.8)
        ax.set_axisbelow(True)


def heatmap(ax: plt.Axes, values: np.ndarray, title: str, cmap: str, show_y: bool) -> None:
    image = ax.imshow(values, cmap=cmap, aspect="auto")
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    threshold = low + .62 * (high - low)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if values[row, column] >= threshold else "#222222"
            ax.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center",
                    fontsize=14.5, fontweight="bold", color=color)
    ax.set_xticks(range(4), MODELS, rotation=24, ha="right")
    ax.set_yticks(range(7), REGIONS if show_y else [""] * 7)
    ax.set_xlabel("Forecast model")
    if show_y:
        ax.set_ylabel("Region")
    ax.set_title(title, fontsize=17, fontweight="bold", pad=6)
    ax.tick_params(axis="x", labelsize=15.5)
    ax.tick_params(axis="y", labelsize=15)
    style_axis(ax)
    image.set_clim(low, high)


def panel_a(fig: plt.Figure, spec, annual: pd.DataFrame) -> None:
    grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=.25)
    region_model = annual.groupby(["region", "model"], observed=True).sum(numeric_only=True)
    cost = np.array([[region_model.loc[(region, model), "balancing_cost_million_cny"] / 1e3
                      for model in MODELS] for region in REGIONS])
    co2 = np.array([[region_model.loc[(region, model), "direct_co2_penalty_mt"]
                     for model in MODELS] for region in REGIONS])
    left = fig.add_subplot(grid[0])
    right = fig.add_subplot(grid[1])
    heatmap(left, cost, "Annual balancing cost\n(billion CNY yr$^{-1}$)", "Blues", True)
    heatmap(right, co2, "Annual direct CO$_2$ penalty\n(Mt yr$^{-1}$)", "GnBu", False)
    left.text(-.17, 1.10, "a", transform=left.transAxes, fontsize=17, fontweight="bold")


def panel_b(fig: plt.Figure, spec, seasonal: pd.DataFrame) -> None:
    ax = fig.add_subplot(spec)
    x = np.arange(4, dtype=float)
    values = [seasonal.loc[seasonal.season == season, "cost_intensity_m_cny_per_gwh"].to_numpy()
              for season in SEASONS]
    bp = ax.boxplot(values, positions=x, widths=.52, patch_artist=True, showfliers=False,
                    medianprops={"color": "#333333", "linewidth": 1.3},
                    whiskerprops={"color": "#777777"}, capprops={"color": "#777777"})
    for box, season in zip(bp["boxes"], SEASONS):
        box.set(facecolor=SEASON_COLORS[season], edgecolor="#999999", linewidth=1.1)

    rng = np.random.default_rng(42)
    means = []
    for season_index, season in enumerate(SEASONS):
        subset = seasonal.loc[seasonal.season == season]
        for _, row in subset.iterrows():
            ax.scatter(season_index + rng.uniform(-.13, .13), row.cost_intensity_m_cny_per_gwh,
                       s=35, color=REGION_COLORS[row.region], edgecolor="white", linewidth=.4,
                       alpha=.78, zorder=4)
        means.append(subset.cost_intensity_m_cny_per_gwh.mean())
    ax.plot(x, means, "o-", color="#D62728", markerfacecolor="black", markeredgecolor="black",
            linewidth=1.3, markersize=7, zorder=5)
    ax.set_xticks(x, [SEASON_SHORT[s] for s in SEASONS])
    ax.set_ylabel(r"Cost intensity (M CNY GWh$^{-1}$)")
    ax.set_title("Seasonal distribution of regional balancing-cost intensity", fontweight="bold", pad=4)
    region_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=REGION_COLORS[r],
                             markeredgecolor="none", label=r, markersize=7) for r in REGIONS]
    mean_handle = Line2D([0], [0], marker="o", color="#D62728", markerfacecolor="black",
                         label="Mean", markersize=8)
    ax.legend(handles=region_handles + [mean_handle], loc="upper left", frameon=False, ncol=4,
              fontsize=13.5, handlelength=1.0, handletextpad=.35, columnspacing=.7)
    ax.tick_params(axis="x", labelsize=16)
    style_axis(ax, grid=True)
    ax.text(-.13, 1.13, "b", transform=ax.transAxes, fontsize=17, fontweight="bold")


def panel_c(fig: plt.Figure, spec, seasonal: pd.DataFrame) -> None:
    ax = fig.add_subplot(spec)
    x = np.arange(len(REGIONS), dtype=float)
    width = .19
    offsets = np.linspace(-1.5 * width, 1.5 * width, 4)
    marker_list = ("o", "s", "D", "^")
    for season_index, season in enumerate(SEASONS):
        heights, low, high = [], [], []
        for region in REGIONS:
            vals = seasonal.loc[(seasonal.season == season) & (seasonal.region == region),
                                "balancing_cost_million_cny"].to_numpy()
            heights.append(vals.mean()); low.append(vals.mean() - vals.min()); high.append(vals.max() - vals.mean())
        positions = x + offsets[season_index]
        ax.bar(positions, heights, width, color=SEASON_COLORS[season], edgecolor="none", label=season,
               yerr=np.array([low, high]), error_kw={"ecolor": "#777777", "capsize": 2, "lw": 1})
        for model_index, model in enumerate(MODELS):
            vals = [seasonal.loc[(seasonal.season == season) & (seasonal.region == region) &
                                 (seasonal.model == model), "balancing_cost_million_cny"].iat[0]
                    for region in REGIONS]
            ax.scatter(positions, vals, marker=marker_list[model_index], s=25,
                       color=MODEL_COLORS[model], edgecolor="white", linewidth=.35, zorder=4)
    ax.set_xticks(x, REGIONS)
    ax.set_xlabel("Region")
    ax.set_ylabel(r"Allocated balancing cost (M CNY season$^{-1}$)")
    ax.set_title("Seasonal regional allocation diagnostic", fontweight="bold", pad=4)
    season_handles = [Patch(facecolor=SEASON_COLORS[s], label=s) for s in SEASONS]
    model_handles = [Line2D([0], [0], marker=marker_list[i], color="none",
                            markerfacecolor=MODEL_COLORS[m], markeredgecolor="none", label=m, markersize=6)
                     for i, m in enumerate(MODELS)]
    ax.legend(handles=season_handles + model_handles, loc="upper right", frameon=False, ncol=2,
              fontsize=13.5, columnspacing=.85, handlelength=1, handletextpad=.35)
    ax.tick_params(axis="x", labelsize=16)
    style_axis(ax, grid=True)
    ax.text(-.13, 1.11, "c", transform=ax.transAxes, fontsize=17, fontweight="bold")


def panel_d(fig: plt.Figure, spec, annual: pd.DataFrame) -> None:
    ax = fig.add_subplot(spec)
    data = annual.copy()
    data["normalized_impact"] = data.balancing_cost_million_cny / data.absolute_error_gwh
    summary = data.groupby(["province", "region"], as_index=False).agg(
        mean_impact=("normalized_impact", "mean"),
        min_impact=("normalized_impact", "min"),
        max_impact=("normalized_impact", "max"),
        mean_error=("absolute_error_gwh", "mean"),
        mean_abs_cost=("balancing_cost_million_cny", lambda x: np.abs(x).mean()),
    ).sort_values("mean_impact", ascending=True).reset_index(drop=True)
    bottom_count = max(1, int(np.floor(.10 * len(summary))))
    flagged = summary.index.isin(summary.nsmallest(bottom_count, "mean_error").index)
    colors = np.where(flagged, "#F28E2B", "#9E9E9E")
    y = np.arange(len(summary))
    error = np.vstack([summary.mean_impact - summary.min_impact,
                       summary.max_impact - summary.mean_impact])
    ax.barh(y, summary.mean_impact, color=colors, edgecolor="none", height=.68,
            xerr=error, error_kw={"ecolor": "#BDBDBD", "capsize": 0, "lw": 1})
    ax.set_yticks(y, [PROVINCE_ABBR[p] for p in summary.province])
    ax.set_xlabel(r"Allocated balancing-cost intensity (M CNY GWh$^{-1}$)")
    ax.set_title("Provincial error-weighted balancing-cost diagnostic",
                 fontweight="bold", pad=4)
    ax.legend(handles=[Patch(facecolor="#F28E2B", label="Lowest 10% error denominator"),
                       Patch(facecolor="#9E9E9E", label="Other provinces")],
              loc="upper left", bbox_to_anchor=(.01, .99), frameon=False, fontsize=13.5)
    ax.tick_params(axis="x", labelsize=15.5)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(-.6, len(summary) + 10.0)
    ax.set_xlim(0, .04)
    ax.set_xticks(np.arange(0, .041, .01))
    style_axis(ax)
    ax.grid(axis="x", color="#C8CED3", linestyle=":", linewidth=.65, alpha=.8)
    ax.set_axisbelow(True)
    ax.text(-.14, 1.08, "d", transform=ax.transAxes, fontsize=17, fontweight="bold")
    # Donut inset: regional composition of mean absolute attributed cost.
    shares = summary.groupby("region").mean_abs_cost.sum().reindex(REGIONS)
    inset = ax.inset_axes([.58, .75, .38, .23])
    inset.patch.set_facecolor("white")
    inset.patch.set_alpha(.94)
    inset.pie(shares, labels=REGIONS, colors=[REGION_COLORS[r] for r in REGIONS], startangle=90,
              counterclock=False, autopct=lambda p: f"{p:.0f}%", pctdistance=.78,
              labeldistance=1.04, radius=.78,
              wedgeprops={"width": .38, "edgecolor": "white"},
              textprops={"fontsize": 11.5, "fontweight": "bold"})
    inset.text(0, 0, "Regional\nshare", ha="center", va="center", fontsize=13, fontweight="bold")


def save_metrics(annual: pd.DataFrame, seasonal: pd.DataFrame, path: Path) -> None:
    annual_output = annual.copy()
    annual_output.insert(0, "table", "province_annual")
    seasonal_output = seasonal.copy()
    seasonal_output.insert(0, "table", "region_season")
    output = pd.concat([annual_output, seasonal_output], ignore_index=True, sort=False)
    output.insert(1, "attribution_method", "hourly provincial absolute-error share")
    output.insert(2, "analysis_period", "8760-hour operational year; leap day excluded")
    output.to_csv(
        path, index=False, float_format="%.6f"
    )


def make_figure(annual: pd.DataFrame, seasonal: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 13.5, "axes.labelsize": 15.5,
        "axes.titlesize": 16, "xtick.labelsize": 15, "ytick.labelsize": 13.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(15.5, 17.0), facecolor="white")
    outer = GridSpec(3, 2, figure=fig, height_ratios=[1.0, .78, 1.0], width_ratios=[1.12, .96],
                     hspace=.36, wspace=.28, left=.075, right=.985, top=.97, bottom=.065)
    panel_a(fig, outer[0, :], annual)
    panel_b(fig, outer[1, 0], seasonal)
    panel_c(fig, outer[2, 0], seasonal)
    panel_d(fig, outer[1:, 1], annual)
    return fig


def main() -> None:
    annual, seasonal = calculate_spatial_metrics()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS / "02_spatial_metrics.csv"
    png_path = FIGURES / "02_spatial.png"
    pdf_path = FIGURES / "02_spatial.pdf"
    save_metrics(annual, seasonal, metrics_path)
    figure = make_figure(annual, seasonal)
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"province annual rows: {len(annual)}")
    print(f"region-season rows: {len(seasonal)}")
    print(metrics_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()

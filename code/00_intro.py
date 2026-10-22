"""Create the introductory seven-day forecast overview (Fig. 00)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from scipy.io import netcdf_file
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parent
FORECAST_FILE = ROOT / "results" / "wind_forecast_2020_all_models.csv"
OBSERVED_FILE = ROOT / "dataset" / "provincial_hourly_2018_2023" / "provincial_hourly_wind_2018_2023.csv"
DISPATCH_FILE = ROOT / "results" / "pypsa_dispatch_2020_all_models.csv"
PERFECT_FILE = ROOT / "results" / "pypsa_two_stage_2020" / "perfect_forecast_hourly.csv"
BOUNDARY_DIR = ROOT / "dataset" / "boundaries" / "python_chinamap"
FIGURE_DIR = ROOT / "figure"

START = pd.Timestamp("2020-09-04 09:00:00")
HOURS = 7 * 24
MODELS = ["CNN", "LSTM", "Transformer", "DNN"]
MODEL_COLORS = {
    "CNN": "#3E8ED0",
    "LSTM": "#20A486",
    "Transformer": "#F39C3D",
    "DNN": "#9C5DD4",
}
OBSERVED_COLOR = "#374151"

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

PROVINCE_REGION = {
    "Beijing": "N", "Tianjin": "N", "Hebei": "N", "Shanxi": "N", "InnerMongolia": "N",
    "Shanghai": "E", "Jiangsu": "E", "Zhejiang": "E", "Anhui": "E", "Fujian": "E",
    "Jiangxi": "E", "Shandong": "E", "Shaanxi": "NW", "Gansu": "NW",
    "Qinghai": "NW", "Ningxia": "NW", "Xinjiang": "NW", "Liaoning": "NE",
    "Jilin": "NE", "Heilongjiang": "NE", "Chongqing": "SW", "Sichuan": "SW",
    "Guizhou": "SW", "Yunnan": "SW", "Tibet": "SW", "Henan": "C",
    "Hubei": "C", "Hunan": "C", "Guangdong": "S", "Guangxi": "S", "Hainan": "S",
}

REGION_BACKGROUNDS = {
    "N": "#F4EDE8", "E": "#EEF3F7", "NW": "#F7F1DF", "NE": "#E8F1F5",
    "SW": "#F2EAF4", "C": "#EEF4EA", "S": "#F7ECE8",
}

TECHNOLOGIES = ["Wind", "Solar", "Hydro", "Nuclear", "Coal", "Gas"]
TECH_COLORS = {
    "Wind": "#59C3BE", "Solar": "#F6D66F", "Hydro": "#91BCDA",
    "Nuclear": "#C7B6DF", "Coal": "#E98A65", "Gas": "#F4B9AD",
}
CAPACITY_FILES = ROOT / "dataset" / "pypsa_china_thu_existing"
CAPACITY_TARGETS_GW = {
    "Wind": 281.530, "Solar": 253.430, "Hydro": 370.160,
    "Nuclear": 49.890, "CoalGas": 1245.170,
}

# Seven-column stepped cartogram matching the supplied reference layout.
# Hong Kong, Macao and Taiwan cells are intentionally left empty because the
# operational model contains 31 mainland provincial nodes only.
TILE_POSITIONS = {
    "Heilongjiang": (0, 5), "Jilin": (0, 6),
    "Xinjiang": (1, 0), "Gansu": (1, 1), "InnerMongolia": (1, 3),
    "Beijing": (1, 4), "Tianjin": (1, 5), "Liaoning": (1, 6),
    "Qinghai": (2, 0), "Ningxia": (2, 1), "Shaanxi": (2, 2),
    "Shanxi": (2, 3), "Hebei": (2, 4), "Shandong": (2, 5),
    "Tibet": (3, 0), "Sichuan": (3, 1), "Chongqing": (3, 2),
    "Hubei": (3, 3), "Henan": (3, 4), "Jiangsu": (3, 5), "Shanghai": (3, 6),
    "Yunnan": (4, 1), "Guizhou": (4, 2), "Hunan": (4, 3),
    "Anhui": (4, 4), "Zhejiang": (4, 5),
    "Guangxi": (5, 2), "Guangdong": (5, 3), "Jiangxi": (5, 4), "Fujian": (5, 5),
    "Hainan": (6, 3),
}

# Reference-layout cells that are shown for geographic completeness but are not
# represented by nodes in the 31-province power-system model.
EMPTY_TILE_POSITIONS = {
    "TW": (4, 6), "MO": (6, 4), "HK": (6, 5),
}


def read_boundaries(path: Path) -> list[np.ndarray]:
    with netcdf_file(path, mode="r", mmap=False) as dataset:
        x = np.asarray(dataset.variables["x"][:], dtype=float)
        y = np.asarray(dataset.variables["y"][:], dtype=float)
        segments = np.asarray(dataset.variables["segments"][:], dtype=int)
    return [
        np.column_stack((x[start:start + count], y[start:start + count]))
        for start, count in segments
        if count >= 2
    ]


def load_period() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    forecast = pd.read_csv(
        FORECAST_FILE,
        parse_dates=["forecast_origin_beijing", "valid_time_beijing"],
    )
    forecast["model"] = forecast["model"].str.upper().replace({"TRANSFORMER": "Transformer"})
    end = START + pd.Timedelta(hours=HOURS - 1)
    forecast = forecast.loc[
        forecast.model.isin(MODELS)
        & forecast.valid_time_beijing.between(START, end)
    ].copy()
    provinces = [
        column for column in forecast.columns
        if column not in {"model", "forecast_origin_beijing", "lead_hour", "valid_time_beijing"}
    ]
    observed = pd.read_csv(OBSERVED_FILE, parse_dates=["timestamp_beijing"])
    observed = observed.loc[
        observed.timestamp_beijing.between(START, end), ["timestamp_beijing", *provinces]
    ].reset_index(drop=True)
    expected = pd.date_range(START, periods=HOURS, freq="h")
    if not observed.timestamp_beijing.equals(pd.Series(expected, name="timestamp_beijing")):
        raise ValueError("Observed seven-day chronology is incomplete")
    for model in MODELS:
        times = forecast.loc[forecast.model == model, "valid_time_beijing"].reset_index(drop=True)
        if not times.equals(pd.Series(expected, name="valid_time_beijing")):
            raise ValueError(f"Forecast chronology is incomplete for {model}")
    if set(provinces) != set(PROVINCE_ABBR):
        raise ValueError("Province columns do not match the 31-province layout")
    return forecast, observed, provinces


def make_source_data(
    forecast: pd.DataFrame,
    observed: pd.DataFrame,
    provinces: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed_indexed = observed.set_index("timestamp_beijing")
    hourly_rows = []
    daily_rows = []
    error_rows = []
    for model in MODELS:
        subset = forecast.loc[forecast.model == model].set_index("valid_time_beijing").sort_index()
        for hour, timestamp in enumerate(subset.index):
            hourly_rows.append({
                "model": model,
                "timestamp_beijing": timestamp,
                "elapsed_hour": hour,
                "observed_gw": observed_indexed.loc[timestamp, provinces].sum() / 1000.0,
                "forecast_gw": subset.loc[timestamp, provinces].sum() / 1000.0,
            })
        for day in range(7):
            day_times = subset.index[day * 24:(day + 1) * 24]
            for province in provinces:
                actual = observed_indexed.loc[day_times, province].to_numpy(dtype=float) / 1000.0
                predicted = subset.loc[day_times, province].to_numpy(dtype=float) / 1000.0
                daily_rows.append({
                    "model": model, "province": province, "day": day + 1,
                    "date": (START + pd.Timedelta(days=day)).date(),
                    "observed_gw": float(actual.mean()),
                    "forecast_gw": float(predicted.mean()),
                })
                error_rows.append({
                    "model": model, "province": province, "day": day + 1,
                    "mae_gw": float(np.mean(np.abs(predicted - actual))),
                })
    return pd.DataFrame(hourly_rows), pd.DataFrame(daily_rows), pd.DataFrame(error_rows)


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.0)
    ax.tick_params(width=0.9, length=4)
    ax.grid(True, color="#E7EBEF", lw=0.75, zorder=0)


def add_panel_label(
    ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.08, fontsize: float = 23,
) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize, fontweight="bold", va="top")


def load_system_effects() -> tuple[pd.DataFrame, pd.DataFrame]:
    dispatch = pd.read_csv(DISPATCH_FILE, parse_dates=["timestamp_beijing"])
    perfect = pd.read_csv(PERFECT_FILE, parse_dates=["timestamp_beijing"])
    perfect = perfect.set_index("timestamp_beijing")["total_operating_cost_cny"]
    dispatch["perfect_total_cost_cny"] = dispatch.timestamp_beijing.map(perfect)
    dispatch["total_cost_difference_cny"] = (
        dispatch.total_operating_cost_cny - dispatch.perfect_total_cost_cny
    )
    dispatch = dispatch.sort_values(["model", "timestamp_beijing"]).reset_index(drop=True)
    dispatch["day_id"] = dispatch.groupby("model").cumcount() // 24
    daily = dispatch.groupby(["model", "day_id"], as_index=False).agg(
        balancing_cost_m_cny=("balancing_cost_cny", lambda values: values.sum() / 1e6),
        total_cost_difference_m_cny=("total_cost_difference_cny", lambda values: values.sum() / 1e6),
    )
    annual = daily.groupby("model", as_index=False).agg(
        balancing_cost_m_cny=("balancing_cost_m_cny", "sum"),
        total_cost_difference_m_cny=("total_cost_difference_m_cny", "sum"),
    )
    return dispatch, daily.merge(annual, on="model", suffixes=("_daily", "_annual"))


def read_capacity_file(name: str, provinces: list[str]) -> pd.Series:
    frame = pd.read_csv(CAPACITY_FILES / name, index_col=0).apply(pd.to_numeric, errors="coerce")
    year_columns = [column for column in frame.columns if str(column).isdigit() and int(column) <= 2020]
    values = frame[year_columns].sum(axis=1) if year_columns else frame.sum(axis=1)
    return values.reindex(provinces).fillna(0.0)


def load_capacity_mix(provinces: list[str]) -> pd.DataFrame:
    capacity = pd.DataFrame(index=provinces)
    capacity["Wind"] = read_capacity_file("onwind_capacity.csv", provinces)
    capacity["Solar"] = read_capacity_file("solar_capacity.csv", provinces)
    capacity["Hydro"] = read_capacity_file("hydroelectricity_capacity.csv", provinces)
    capacity["Nuclear"] = read_capacity_file("nuclear_capacity.csv", provinces)
    capacity["Coal"] = (
        read_capacity_file("coal_capacity.csv", provinces)
        + read_capacity_file("chp_coal_capacity.csv", provinces)
    )
    capacity["Gas"] = (
        read_capacity_file("ocgt_capacity.csv", provinces)
        + read_capacity_file("chp_gas_capacity.csv", provinces)
    )
    for technology in ("Wind", "Solar", "Hydro", "Nuclear"):
        total = capacity[technology].sum()
        if total > 0:
            capacity[technology] *= CAPACITY_TARGETS_GW[technology] / total
    conventional_total = capacity[["Coal", "Gas"]].to_numpy().sum()
    if conventional_total > 0:
        capacity[["Coal", "Gas"]] *= CAPACITY_TARGETS_GW["CoalGas"] / conventional_total
    capacity.index.name = "province"
    return capacity.reset_index()


def plot_forecast_error(ax: plt.Axes, dispatch: pd.DataFrame) -> None:
    rng = np.random.default_rng(42)
    positions = np.arange(len(MODELS))
    datasets = [
        dispatch.loc[dispatch.model == model, "forecast_error_mw"].to_numpy(dtype=float) / 1000.0
        for model in MODELS
    ]
    violin = ax.violinplot(datasets, positions=positions + 0.16, widths=0.72,
                           showmeans=False, showmedians=False, showextrema=False)
    for body, model in zip(violin["bodies"], MODELS):
        vertices = body.get_paths()[0].vertices
        vertices[:, 0] = np.maximum(vertices[:, 0], positions[MODELS.index(model)] + 0.16)
        body.set_facecolor(MODEL_COLORS[model])
        body.set_edgecolor("none")
        body.set_alpha(0.24)
    for index, (model, values) in enumerate(zip(MODELS, datasets)):
        selected = rng.choice(len(values), size=min(180, len(values)), replace=False)
        jitter = rng.normal(-0.17, 0.035, len(selected))
        ax.scatter(np.full(len(selected), index) + jitter, values[selected], s=9,
                   color=MODEL_COLORS[model], alpha=0.45, edgecolors="none", rasterized=True)
    boxes = ax.boxplot(datasets, positions=positions, widths=0.25, patch_artist=True,
                       showfliers=False, medianprops={"color": "white", "lw": 1.5})
    for patch, model in zip(boxes["boxes"], MODELS):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_edgecolor(MODEL_COLORS[model])
        patch.set_alpha(0.92)
    for line in boxes["whiskers"] + boxes["caps"]:
        line.set_color("#59636E")
    ax.axhline(0, color="#6B7280", lw=0.9, ls="--")
    ax.set_xticks(positions, MODELS, rotation=17, ha="right", fontsize=14.0)
    ax.set_ylabel("Wind-power forecast error (GW)", fontsize=16.0)
    ax.set_xlabel("Forecast model", fontsize=16.0)
    style_axis(ax)
    ax.tick_params(axis="y", labelsize=14.0)
    add_panel_label(ax, "a", x=-0.12, fontsize=27)


def plot_annual_bars(ax: plt.Axes, effects: pd.DataFrame, column: str, title: str) -> None:
    values = [effects.loc[effects.model == model, column].iat[0] for model in MODELS]
    xpos = np.arange(len(MODELS))
    bars = ax.bar(xpos, values, width=0.68, color=[MODEL_COLORS[model] for model in MODELS], zorder=2)
    offset = 0.025 * max(values)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.1f}",
                ha="center", va="bottom", fontsize=10.5)
    ax.set_xticks(xpos, MODELS, rotation=27, ha="right")
    ax.set_ylabel(f"Annual {title.lower()} (million CNY)")
    ax.set_title(title, loc="left", pad=8)
    ax.set_ylim(0, max(values) * 1.14)
    style_axis(ax)


def plot_daily_half_violins(ax: plt.Axes, effects: pd.DataFrame, column: str) -> None:
    positions = np.arange(len(MODELS))[::-1]
    all_values = effects[column].to_numpy(dtype=float)
    low = min(float(np.quantile(all_values, 0.01)), 0.0)
    high = max(float(np.quantile(all_values, 0.99)), 0.0)
    margin = 0.12 * max(high - low, 1.0)
    grid = np.linspace(low - margin, high + margin, 240)
    for position, model in zip(positions, MODELS):
        values = effects.loc[effects.model == model, column].to_numpy(dtype=float)
        density = gaussian_kde(values)(grid)
        density = density / max(density.max(), 1e-12) * 0.30
        ax.fill_between(grid, position, position + density, color=MODEL_COLORS[model], alpha=0.70, linewidth=0)
        ax.plot([np.median(values), np.median(values)], [position + 0.04, position + 0.27],
                color="#222222", lw=1.5)
    ax.axvline(0, color="#888888", lw=0.8, ls="--")
    ax.set_xlim(grid.min(), grid.max())
    ax.set_ylim(-0.15, 3.42)
    ax.set_yticks([])
    ax.set_xlabel("million CNY\nper day")
    ax.set_title("Daily distribution", pad=8)
    style_axis(ax)


def plot_effect_pair(
    ax_bar: plt.Axes,
    ax_daily: plt.Axes,
    effects: pd.DataFrame,
    annual_column: str,
    daily_column: str,
    title: str,
    panel: str,
) -> None:
    plot_annual_bars(ax_bar, effects, annual_column, title)
    plot_daily_half_violins(ax_daily, effects, daily_column)
    # In the stacked 2 x 2 composition, place labels above the y-axis title so
    # the lower-row c label cannot merge with its vertical unit text.
    add_panel_label(ax_bar, panel, x=-0.12, y=1.16)


def draw_map_context(ax: plt.Axes) -> None:
    national = read_boundaries(BOUNDARY_DIR / "cnmap.nc")
    provincial = read_boundaries(BOUNDARY_DIR / "cnhimap.nc")
    ax.add_collection(PolyCollection(
        provincial, facecolors="#C9CDD0", edgecolors="white", lw=0.58, alpha=1.0,
    ))
    ax.add_collection(LineCollection(national, colors="#929AA1", lw=0.85, alpha=1.0))
    ax.set_xlim(73, 135)
    ax.set_ylim(17, 54)
    # The reference uses the cartogram and geographic silhouette in one shared
    # rectangular envelope.  Auto aspect avoids an equal-aspect letterbox that
    # would otherwise detach the edge tiles from the silhouette.
    ax.set_aspect("auto")
    ax.axis("off")


def plot_province_tile(ax: plt.Axes, province: str, capacity: pd.DataFrame) -> None:
    values = capacity.loc[capacity.province == province, TECHNOLOGIES].iloc[0].to_numpy(dtype=float)
    if values.sum() <= 0:
        values = np.ones(len(TECHNOLOGIES), dtype=float)
    # Keep the parent tile rectangular, as in the reference.  Drawing the pie
    # on an inset prevents matplotlib's pie aspect from shrinking the tile and
    # opening artificial gaps between adjacent province cells.
    pie_ax = ax.inset_axes([0.21, 0.16, 0.58, 0.66])
    pie_ax.pie(
        values,
        colors=[TECH_COLORS[technology] for technology in TECHNOLOGIES],
        startangle=90,
        counterclock=False,
        radius=1.0,
        wedgeprops={"edgecolor": "white", "linewidth": 0.45},
    )
    pie_ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)
    ax.patch.set_visible(True)
    # Transparent cells keep the geographic China silhouette visually primary.
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)
    for spine in ax.spines.values():
        spine.set_color("#8D969F")
        spine.set_linewidth(0.75)
        spine.set_linestyle((0, (3, 2)))
    ax.text(0.05, 0.94, PROVINCE_ABBR[province], transform=ax.transAxes,
            ha="left", va="top", fontsize=10.0, color="#252B31")


def plot_empty_tile(ax: plt.Axes, label: str) -> None:
    """Draw the reference cartogram cell without inventing capacity data."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)
    ax.patch.set_visible(True)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)
    for spine in ax.spines.values():
        spine.set_color("#8D969F")
        spine.set_linewidth(0.75)
        spine.set_linestyle((0, (3, 2)))
    ax.text(0.05, 0.94, label, transform=ax.transAxes,
            ha="left", va="top", fontsize=10.0, color="#252B31")


def make_figure(
    capacity: pd.DataFrame,
    dispatch: pd.DataFrame,
    effects: pd.DataFrame,
) -> tuple[Path, Path]:
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 11.5, "axes.titlesize": 15.5,
        "axes.labelsize": 12.5, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(18.4, 15.0), facecolor="white")
    outer = GridSpec(2, 1, figure=fig, height_ratios=[0.43, 0.57], hspace=0.15)

    # Use the same centred horizontal envelope as panel d.  This keeps d fixed
    # and aligns the left/right edges by shrinking only panels a-c.
    top_band = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[0], width_ratios=[0.195, 0.61, 0.195], wspace=0.0,
    )
    # Compact 2 x 2 upper composition: panel a occupies the left column,
    # while b and c form the upper- and lower-right cells.
    top = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=top_band[0, 1], width_ratios=[1.08, 1.0],
        height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.42,
    )
    ax_error = fig.add_subplot(top[:, 0])
    top_b = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=top[0, 1], width_ratios=[1.85, 0.80], wspace=0.24,
    )
    top_c = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=top[1, 1], width_ratios=[1.85, 0.80], wspace=0.24,
    )
    ax_balancing = fig.add_subplot(top_b[0, 0])
    ax_balancing_daily = fig.add_subplot(top_b[0, 1])
    ax_total = fig.add_subplot(top_c[0, 0])
    ax_total_daily = fig.add_subplot(top_c[0, 1])
    plot_forecast_error(ax_error, dispatch)
    plot_effect_pair(
        ax_balancing, ax_balancing_daily, effects,
        "balancing_cost_m_cny_annual", "balancing_cost_m_cny_daily",
        "Balancing cost", "b",
    )
    plot_effect_pair(
        ax_total, ax_total_daily, effects,
        "total_cost_difference_m_cny_annual", "total_cost_difference_m_cny_daily",
        "Total cost difference", "c",
    )

    # The reference cartogram is deliberately compact (roughly 1.2:1 for its
    # seven-by-seven cell matrix), not stretched across the full page width.
    lower_band = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1], width_ratios=[0.195, 0.61, 0.195], wspace=0.0,
    )
    lower_holder = fig.add_subplot(lower_band[0, 1])
    draw_map_context(lower_holder)
    technology_handles = [
        Patch(facecolor=TECH_COLORS[technology], edgecolor="none", label=technology)
        for technology in TECHNOLOGIES
    ]
    lower_holder.text(0.0, 0.235, "Installed-capacity mix\n(share of provincial total)",
                      transform=lower_holder.transAxes, fontsize=11.0,
                      ha="left", va="top")
    lower_holder.legend(
        handles=technology_handles, loc="lower left", bbox_to_anchor=(-0.005, 0.065),
        ncol=2, frameon=False, fontsize=10.3, handlelength=1.15,
        columnspacing=0.9, handletextpad=0.38, borderaxespad=0.0,
    )

    tiles = GridSpecFromSubplotSpec(7, 7, subplot_spec=lower_band[0, 1], wspace=0.025, hspace=0.025)
    for province, (row, column) in TILE_POSITIONS.items():
        ax = fig.add_subplot(tiles[row, column])
        plot_province_tile(ax, province, capacity)
        if province == "Xinjiang":
            ax.text(-0.18, 1.30, "d", transform=ax.transAxes, fontsize=23,
                    fontweight="bold", ha="left", va="top", clip_on=False)
    for label, (row, column) in EMPTY_TILE_POSITIONS.items():
        ax = fig.add_subplot(tiles[row, column])
        plot_empty_tile(ax, label)

    # South China Sea inset: boundaries only and no text.
    inset = lower_holder.inset_axes([0.870, 0.020, 0.105, 0.170], zorder=10)
    inset.add_collection(LineCollection(read_boundaries(BOUNDARY_DIR / "cnhimap.nc"),
                                        colors="#68717A", lw=0.36))
    inset.add_collection(LineCollection(read_boundaries(BOUNDARY_DIR / "cnmap.nc"),
                                        colors="#30363C", lw=0.58))
    inset.set_xlim(105, 125)
    inset.set_ylim(3, 25)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_color("#59636E")
        spine.set_linewidth(0.7)

    fig.subplots_adjust(left=0.050, right=0.988, top=0.970, bottom=0.035)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / "00_intro.png"
    pdf = FIGURE_DIR / "00_intro.pdf"
    fig.savefig(png, dpi=420, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def main() -> None:
    provinces = list(PROVINCE_ABBR)
    dispatch, effects = load_system_effects()
    capacity = load_capacity_mix(provinces)
    forecast_errors = dispatch[["model", "timestamp_beijing", "forecast_error_mw"]].copy()
    forecast_errors["forecast_error_gw"] = forecast_errors["forecast_error_mw"] / 1000.0
    forecast_errors.to_csv(
        FIGURE_DIR / "00_intro_forecast_error_values.csv", index=False, float_format="%.6f"
    )
    effects.to_csv(FIGURE_DIR / "00_intro_system_values.csv", index=False, float_format="%.6f")
    capacity.to_csv(FIGURE_DIR / "00_intro_capacity_values.csv", index=False, float_format="%.6f")
    png, pdf = make_figure(capacity, dispatch, effects)
    print(f"saved figure: {png}\nsaved figure: {pdf}")


if __name__ == "__main__":
    main()

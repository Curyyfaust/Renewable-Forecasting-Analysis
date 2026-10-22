"""Three spatial supplementary figures derived from the project results.

Outputs
-------
03_provincial_co2
    Annual direct-CO2 attribution by province for the four forecast models.
03_provincial_cost
    Annual balancing-cost attribution by province for the four models.
03_provincial_variance
    Monthly provincial variance of normalized wind-power forecast errors.

The figures retain Matplotlib's default colour cycle and default colormap.
Only font sizes and layouts are adjusted for readability.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "supp" / "supp_figure"
SPATIAL_METRICS = OUTPUT / "03_provincial_attribution_values.csv"
FORECAST = ROOT / "results" / "wind_forecast_2020_all_models.csv"
ACTUAL = ROOT / "dataset" / "provincial_hourly_2018_2023" / "provincial_hourly_wind_2018_2023.csv"
BOUNDARIES = ROOT / "dataset" / "boundaries" / "python_chinamap"

MODELS = ("CNN", "LSTM", "Transformer", "DNN")
MODEL_KEYS = {"CNN": "cnn", "LSTM": "lstm", "Transformer": "transformer", "DNN": "dnn"}
MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
PROVINCE_COORDS = {
    "Anhui": (117.3, 31.9), "Beijing": (116.4, 39.9), "Chongqing": (106.6, 29.6),
    "Fujian": (118.1, 26.1), "Gansu": (103.8, 36.1), "Guangdong": (113.3, 23.1),
    "Guangxi": (108.3, 23.8), "Guizhou": (106.7, 26.6), "Hainan": (110.3, 19.2),
    "Hebei": (114.5, 38.1), "Heilongjiang": (126.6, 45.8), "Henan": (113.7, 34.8),
    "Hubei": (114.3, 30.6), "Hunan": (112.9, 28.2), "InnerMongolia": (111.7, 41.7),
    "Jiangsu": (118.8, 32.1), "Jiangxi": (115.9, 28.7), "Jilin": (125.3, 43.9),
    "Liaoning": (123.4, 41.8), "Ningxia": (106.3, 38.5), "Qinghai": (101.8, 36.6),
    "Shaanxi": (108.9, 34.3), "Shandong": (117.0, 36.7), "Shanghai": (121.5, 31.2),
    "Shanxi": (112.6, 37.9), "Sichuan": (104.1, 30.7), "Tianjin": (117.2, 39.1),
    "Tibet": (91.1, 29.7), "Xinjiang": (87.6, 43.8), "Yunnan": (102.7, 25.0),
    "Zhejiang": (120.2, 30.3),
}


def nc_segments(path: Path) -> list[np.ndarray]:
    """Read polyline segments from the bundled NCL-style NetCDF map."""
    with xr.open_dataset(path, decode_cf=False) as dataset:
        x = dataset.x.values
        y = dataset.y.values
        indices = dataset.segments.values.astype(int)
    return [np.column_stack((x[start:start + count], y[start:start + count]))
            for start, count in indices if count >= 2]


def province_polygons(provincial_segments: list[np.ndarray]) -> dict[str, object]:
    """Polygonize the provincial line network and match provinces by map nodes."""
    lines = [LineString(segment) for segment in provincial_segments]
    polygons = list(polygonize(unary_union(lines)))
    output: dict[str, object] = {}
    for province, coordinate in PROVINCE_COORDS.items():
        matches = [polygon for polygon in polygons if polygon.covers(Point(coordinate))]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one map polygon for {province}, found {len(matches)}")
        output[province] = matches[0]
    return output


def load_annual_metrics() -> pd.DataFrame:
    """Load the province-level attribution table stored with Figure 03."""
    data = pd.read_csv(SPATIAL_METRICS)
    data = data.loc[data.table == "province_annual"].copy()
    expected = {(model, province) for model in MODELS for province in PROVINCE_COORDS}
    found = set(zip(data.model, data.province))
    if found != expected:
        raise ValueError("Province-model coverage in 03_provincial_attribution_values.csv is incomplete")
    return data


def build_monthly_variance() -> pd.DataFrame:
    """Calculate monthly normalized forecast-error variance in percentage points squared."""
    forecast = pd.read_csv(FORECAST, parse_dates=["valid_time_beijing"])
    actual = pd.read_csv(ACTUAL, parse_dates=["timestamp_beijing"]).set_index("timestamp_beijing")
    provinces = list(PROVINCE_COORDS)
    rows: list[dict[str, float | int | str]] = []

    for model in MODELS:
        prediction = (
            forecast.loc[forecast.model == MODEL_KEYS[model]]
            .set_index("valid_time_beijing").sort_index()[provinces]
        )
        observation = actual.reindex(prediction.index)[provinces]
        if prediction.isna().any().any() or observation.isna().any().any():
            raise ValueError(f"Incomplete provincial forecast coverage for {model}")

        # A province-specific available-power scale keeps the variance
        # comparable across provinces without importing an external capacity
        # dataset.  It is calculated from this project's observed and forecast
        # records only and is fixed across all months.
        scale = pd.concat([prediction.max(axis=0), observation.max(axis=0)], axis=1).max(axis=1)
        scale = scale.replace(0.0, np.nan)
        normalized_error = prediction.sub(observation).div(scale, axis=1) * 100.0
        normalized_error["month"] = normalized_error.index.month
        for month, group in normalized_error.groupby("month"):
            variance = group[provinces].var(ddof=0)
            for province in provinces:
                rows.append({
                    "model": model,
                    "province": province,
                    "month": int(month),
                    "forecast_error_variance_pp2": float(variance[province]),
                })
    return pd.DataFrame(rows)


def add_map_panel(
    ax: plt.Axes,
    values: pd.Series,
    polygons: dict[str, object],
    national: list[np.ndarray],
    provincial: list[np.ndarray],
    norm: Normalize,
    cmap,
    title: str,
    panel_label: str,
) -> None:
    """Draw one complete-China province choropleth with a South China Sea inset."""
    patches = []
    facecolors = []
    for province, polygon in polygons.items():
        patches.append(MplPolygon(np.asarray(polygon.exterior.coords), closed=True))
        facecolors.append(cmap(norm(float(values[province]))))
    ax.add_collection(PatchCollection(patches, facecolors=facecolors, edgecolors="none"))
    ax.add_collection(LineCollection(provincial, colors="0.55", linewidths=0.45))
    ax.add_collection(LineCollection(national, colors="0.15", linewidths=0.9))
    ax.set_xlim(73.0, 135.5)
    ax.set_ylim(17.0, 54.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=18)
    ax.text(0.01, 0.98, panel_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=19, fontweight="bold")
    ax.set_xticks([80, 100, 120], [r"80$^\circ$E", r"100$^\circ$E", r"120$^\circ$E"])
    ax.set_yticks([20, 30, 40, 50], [r"20$^\circ$N", r"30$^\circ$N", r"40$^\circ$N", r"50$^\circ$N"])
    ax.tick_params(labelsize=14)

    inset = ax.inset_axes([0.82, 0.03, 0.16, 0.28])
    inset.add_collection(LineCollection(provincial, colors="0.55", linewidths=0.35))
    inset.add_collection(LineCollection(national, colors="0.15", linewidths=0.55))
    inset.set_xlim(105.0, 125.0)
    inset.set_ylim(3.0, 25.0)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])


def make_choropleth(
    annual: pd.DataFrame,
    polygons: dict[str, object],
    national: list[np.ndarray],
    provincial: list[np.ndarray],
    column: str,
    colorbar_label: str,
    cmap_name: str | None = None,
) -> plt.Figure:
    """Create a shared-scale 2 x 2 map for one annual metric."""
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 14})
    values = annual[column].astype(float)
    norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
    cmap = plt.get_cmap(cmap_name)
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    for index, (ax, model) in enumerate(zip(axes.flat, MODELS)):
        series = annual.loc[annual.model == model].set_index("province")[column]
        add_map_panel(ax, series, polygons, national, provincial, norm, cmap,
                      model, chr(ord("a") + index))
        if index % 2:
            ax.tick_params(labelleft=False)
        if index < 2:
            ax.tick_params(labelbottom=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.16,
                        wspace=0.08, hspace=0.12)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar_axis = fig.add_axes([0.20, 0.055, 0.60, 0.025])
    colorbar = fig.colorbar(scalar, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(colorbar_label, fontsize=17)
    colorbar.ax.tick_params(labelsize=14)
    return fig


def make_variance_figure(variance: pd.DataFrame) -> plt.Figure:
    """Create 31 provincial monthly small multiples."""
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 12})
    provinces = sorted(PROVINCE_COORDS)
    fig, axes = plt.subplots(6, 6, figsize=(18, 18), sharex=True)
    for index, province in enumerate(provinces):
        ax = axes.flat[index]
        for model in MODELS:
            values = variance.loc[(variance.province == province) &
                                  (variance.model == model)].sort_values("month")
            ax.plot(values.month, values.forecast_error_variance_pp2, label=model)
        ax.set_title(province, fontsize=13)
        ax.set_xlim(1, 12)
        ax.set_xticks([1, 4, 7, 10], ["Jan", "Apr", "Jul", "Oct"])
        ax.tick_params(labelsize=10)
    for ax in axes.flat[len(provinces):]:
        ax.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=15)
    fig.supylabel(r"Forecast-error variance (percentage points$^2$)", fontsize=17)
    fig.supxlabel("Month", fontsize=17)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.95, bottom=0.06,
                        wspace=0.30, hspace=0.42)
    return fig


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.pdf")
    fig.savefig(OUTPUT / f"{stem}.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    annual = load_annual_metrics()
    variance = build_monthly_variance()
    national = nc_segments(BOUNDARIES / "cnmap.nc")
    provincial = nc_segments(BOUNDARIES / "cnhimap.nc")
    polygons = province_polygons(provincial)

    annual.to_csv(OUTPUT / "03_provincial_attribution_values.csv", index=False)
    variance.to_csv(OUTPUT / "03_provincial_variance_values.csv", index=False)

    co2 = make_choropleth(
        annual, polygons, national, provincial,
        "direct_co2_penalty_mt",
        r"Direct CO$_2$ difference relative to Perfect (Mt)",
        cmap_name="RdYlBu_r",
    )
    save_figure(co2, "03_provincial_co2")

    cost = make_choropleth(
        annual, polygons, national, provincial,
        "balancing_cost_million_cny",
        "Balancing-cost difference relative to Perfect (M CNY)",
        cmap_name="RdYlBu_r",
    )
    save_figure(cost, "03_provincial_cost")

    variance_figure = make_variance_figure(variance)
    save_figure(variance_figure, "03_provincial_variance")

    for stem in ("03_provincial_co2", "03_provincial_cost", "03_provincial_variance"):
        print(f"Saved {OUTPUT / (stem + '.pdf')}")
        print(f"Saved {OUTPUT / (stem + '.png')}")


if __name__ == "__main__":
    main()

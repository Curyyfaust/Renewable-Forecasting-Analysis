"""Plot transmission differences in a longitude-latitude network layout.

The 2 x 4 layout compares four forecast models in the day-ahead and intraday
stages. Province nodes retain their geographic coordinates, but no national
or provincial map boundaries are drawn. Pale links show the shared four-model
mean flow backbone and coloured arrows show model deviations from that mean.
Leap day is excluded, so every annual mean covers 8,760 hours.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pypsa
from scipy.io import netcdf_file


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
POWER = DATA / "provincial_hourly_2018_2023"
CAPACITY = DATA / "pypsa_china_thu_existing"
FORECAST = ROOT / "results" / "wind_forecast_2020_all_models.csv"
EDGES = ROOT / "PyPSA-china" / "resources" / "data" / "grids" / "edges_current.csv"
BOUNDARY_DIR = ROOT / "dataset" / "boundaries" / "python_chinamap"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figure"

MODELS = ("CNN", "LSTM", "Transformer", "DNN")
MODEL_KEYS = {"CNN": "cnn", "LSTM": "lstm", "Transformer": "transformer", "DNN": "dnn"}
MODEL_COLORS = {"CNN": "#1787BE", "LSTM": "#F29B00", "Transformer": "#9567CC", "DNN": "#00A37A"}

# Costs and capacity calibration are kept identical to the two-stage
# experiment in PyPSA-china/main_dispatch.py.
MARGINAL_COST = {"hydro": 20.0, "nuclear": 60.0, "coal": 280.0, "gas": 550.0}
UP_COST = {"hydro": 30.0, "nuclear": 120.0, "coal": 80.0, "gas": 120.0}
DOWN_COST = {"hydro": 15.0, "nuclear": 60.0, "coal": 40.0, "gas": 60.0}
TARGET_CAPACITY = {
    "wind": 281_530.0,
    "solar": 253_430.0,
    "hydro": 370_160.0,
    "nuclear": 49_890.0,
    "coal_gas": 1_245_170.0,
}

# Geographic label/arrow anchors (degrees east/north).  The electricity model
# has 31 mainland provincial nodes; Taiwan is retained in the map boundary but
# is not fabricated as a network node.
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


def read_power(name: str) -> pd.DataFrame:
    """Read one complete provincial hourly power table."""
    path = POWER / f"provincial_hourly_{name}_2018_2023.csv"
    frame = pd.read_csv(path, parse_dates=["timestamp_beijing"])
    return frame.set_index("timestamp_beijing").sort_index().astype(float)


def read_capacity_file(name: str, provinces: list[str]) -> pd.Series:
    frame = pd.read_csv(CAPACITY / name, index_col=0).apply(pd.to_numeric, errors="coerce")
    year_columns = [column for column in frame.columns if str(column).isdigit() and int(column) <= 2020]
    values = frame[year_columns].sum(axis=1) if year_columns else frame.sum(axis=1)
    return values.reindex(provinces).fillna(0.0)


def load_capacities(provinces: list[str]) -> pd.DataFrame:
    capacities = pd.DataFrame(index=provinces)
    capacities["wind"] = read_capacity_file("onwind_capacity.csv", provinces)
    capacities["solar"] = read_capacity_file("solar_capacity.csv", provinces)
    capacities["hydro"] = read_capacity_file("hydroelectricity_capacity.csv", provinces)
    capacities["nuclear"] = read_capacity_file("nuclear_capacity.csv", provinces)
    capacities["coal"] = read_capacity_file("coal_capacity.csv", provinces) + read_capacity_file("chp_coal_capacity.csv", provinces)
    capacities["gas"] = read_capacity_file("ocgt_capacity.csv", provinces) + read_capacity_file("chp_gas_capacity.csv", provinces)
    for carrier in ("wind", "solar", "hydro", "nuclear"):
        capacities[carrier] *= TARGET_CAPACITY[carrier] / capacities[carrier].sum()
    capacities[["coal", "gas"]] *= TARGET_CAPACITY["coal_gas"] / capacities[["coal", "gas"]].to_numpy().sum()
    return capacities


def load_inputs() -> dict[str, object]:
    """Load the complete 8,760-hour 2020 experiment."""
    forecast = pd.read_csv(FORECAST, parse_dates=["forecast_origin_beijing", "valid_time_beijing"])
    metadata = {"model", "forecast_origin_beijing", "lead_hour", "valid_time_beijing"}
    provinces = [column for column in forecast.columns if column not in metadata]
    if set(provinces) != set(PROVINCE_COORDS):
        raise ValueError("Forecast provinces do not match the 31 map/network nodes")

    all_times = pd.DatetimeIndex(sorted(forecast.valid_time_beijing.unique()))
    times = all_times[~((all_times.month == 2) & (all_times.day == 29))]
    if len(times) != 8_760:
        raise ValueError(f"Expected 8,760 non-leap-day hours, found {len(times)}")
    wind = read_power("wind").reindex(times)[provinces]
    solar = read_power("solar").reindex(times)[provinces]
    load = read_power("load").reindex(times)[provinces]
    if wind.isna().any().any() or solar.isna().any().any() or load.isna().any().any():
        raise ValueError("Observed wind, solar or load is incomplete for the analysis year")
    scenarios = {
        model: forecast.loc[forecast.model == key].set_index("valid_time_beijing").reindex(times)[provinces].clip(lower=0.0)
        for model, key in MODEL_KEYS.items()
    }
    if any(frame.isna().any().any() for frame in scenarios.values()):
        raise ValueError("At least one model forecast is incomplete for the analysis year")
    capacities = load_capacities(provinces)
    for province in provinces:
        capacities.at[province, "wind"] = max(
            capacities.at[province, "wind"],
            1.001 * max(wind[province].max(), *(values[province].max() for values in scenarios.values())),
        )
        capacities.at[province, "solar"] = max(capacities.at[province, "solar"], 1.001 * solar[province].max())

    edges = pd.read_csv(EDGES, header=None, names=["bus0", "bus1", "p_nom"])
    edges = edges.query("bus0 in @provinces and bus1 in @provinces").reset_index(drop=True)
    return {
        "times": times,
        "provinces": provinces,
        "observed": wind,
        "solar": solar,
        "load": load,
        "scenarios": scenarios,
        "capacities": capacities,
        "edges": edges,
    }


def base_network(times: pd.DatetimeIndex, provinces: list[str], load: pd.DataFrame, edges: pd.DataFrame) -> pypsa.Network:
    network = pypsa.Network()
    network.set_snapshots(times)
    network.snapshot_weightings.loc[:, :] = 1.0
    carriers = ["AC", "transmission", "wind", "solar", "hydro", "nuclear", "coal", "gas", "shed"]
    carriers += [f"{direction}_{carrier}" for direction in ("base", "up", "down") for carrier in MARGINAL_COST]
    for carrier in carriers:
        network.add("Carrier", carrier)
    for province in provinces:
        network.add("Bus", province, carrier="AC")
        network.add("Load", f"load::{province}", bus=province, p_set=load[province])
    for index, edge in edges.iterrows():
        network.add(
            "Link", f"line::{index}", bus0=edge.bus0, bus1=edge.bus1,
            carrier="transmission", p_nom=float(edge.p_nom), p_min_pu=-1.0, efficiency=1.0,
        )
    return network


def day_ahead_network(data: dict[str, object], times: pd.DatetimeIndex, forecast: pd.DataFrame) -> pypsa.Network:
    network = base_network(times, data["provinces"], data["load"].loc[times], data["edges"])
    capacities = data["capacities"]
    for province in data["provinces"]:
        for carrier, profile in (("wind", forecast), ("solar", data["solar"].loc[times])):
            capacity = float(capacities.at[province, carrier])
            network.add(
                "Generator", f"{carrier}::{province}", bus=province, carrier=carrier,
                p_nom=capacity, marginal_cost=0.0,
                p_max_pu=(profile[province] / capacity).clip(0.0, 1.0),
            )
        for carrier, cost in MARGINAL_COST.items():
            capacity = float(capacities.at[province, carrier])
            if capacity > 0.0:
                network.add("Generator", f"{carrier}::{province}", bus=province, carrier=carrier, p_nom=capacity, marginal_cost=cost)
        network.add("Generator", f"shed::{province}", bus=province, carrier="shed", p_nom=2.0 * float(data["load"].loc[times, province].max()), marginal_cost=10_000.0)
    return network


def intraday_network(data: dict[str, object], times: pd.DatetimeIndex, day_ahead: pypsa.Network) -> pypsa.Network:
    network = base_network(times, data["provinces"], data["load"].loc[times], data["edges"])
    capacities = data["capacities"]
    for province in data["provinces"]:
        for carrier, profile, offer in (("wind", data["observed"].loc[times], -200.0), ("solar", data["solar"].loc[times], -150.0)):
            capacity = float(capacities.at[province, carrier])
            network.add(
                "Generator", f"{carrier}::{province}", bus=province, carrier=carrier,
                p_nom=capacity, marginal_cost=offer,
                p_max_pu=(profile[province] / capacity).clip(0.0, 1.0),
            )
        for carrier in MARGINAL_COST:
            old_name = f"{carrier}::{province}"
            if old_name not in day_ahead.generators.index:
                continue
            capacity = float(day_ahead.generators.at[old_name, "p_nom"])
            schedule = day_ahead.generators_t.p[old_name].clip(0.0, capacity)
            network.add(
                "Generator", f"base::{old_name}", bus=province, carrier=f"base_{carrier}",
                p_nom=capacity, marginal_cost=0.0,
                p_min_pu=schedule / capacity, p_max_pu=schedule / capacity,
            )
            network.add(
                "Generator", f"up::{old_name}", bus=province, carrier=f"up_{carrier}",
                p_nom=capacity, marginal_cost=UP_COST[carrier], p_max_pu=(capacity - schedule) / capacity,
            )
            network.add(
                "Generator", f"down::{old_name}", bus=province, carrier=f"down_{carrier}",
                p_nom=capacity, marginal_cost=-DOWN_COST[carrier],
                p_min_pu=-schedule / capacity, p_max_pu=0.0,
            )
        network.add("Generator", f"shed::{province}", bus=province, carrier="shed", p_nom=2.0 * float(data["load"].loc[times, province].max()), marginal_cost=10_000.0)
    return network


def solve(network: pypsa.Network, solver: str) -> None:
    status, condition = network.optimize(solver_name=solver, log_to_console=False)
    if status != "ok":
        raise RuntimeError(f"PyPSA solve failed: {status}, {condition}")


def summarize_flows(flow: pd.DataFrame, edges: pd.DataFrame, model: str, stage: str) -> pd.DataFrame:
    """Summarize hourly signed corridor flows into annual statistics."""
    rows = []
    for index, edge in edges.iterrows():
        values = flow[f"line::{index}"]
        mean_flow = float(values.mean())
        rows.append({
            "model": model,
            "stage": stage,
            "hours": len(values),
            "line": f"line::{index}",
            "bus0": edge.bus0,
            "bus1": edge.bus1,
            "capacity_mw": float(edge.p_nom),
            "mean_flow_mw": mean_flow,
            "mean_abs_flow_mw": float(values.abs().mean()),
            "mean_utilization_pct": 100.0 * float(values.abs().mean()) / float(edge.p_nom),
            "p05_flow_mw": float(values.quantile(0.05)),
            "p95_flow_mw": float(values.quantile(0.95)),
        })
    return pd.DataFrame(rows)


def run_dispatch(data: dict[str, object], solver: str) -> pd.DataFrame:
    """Solve monthly chunks and return 8,760-hour mean line flows."""
    tables = []
    month_chunks = [pd.DatetimeIndex(group.to_numpy()) for _, group in pd.Series(data["times"], index=data["times"]).groupby(data["times"].to_period("M"))]
    total_chunks = len(month_chunks)
    for model in MODELS:
        stage_flows = {"Day-ahead": [], "Intraday": []}
        for chunk_number, times in enumerate(month_chunks, start=1):
            print(f"{model} block {chunk_number:02d}/{total_chunks:02d}: day-ahead", flush=True)
            day_ahead = day_ahead_network(data, times, data["scenarios"][model].loc[times])
            solve(day_ahead, solver)
            stage_flows["Day-ahead"].append(day_ahead.links_t.p0.copy())
            print(f"{model} block {chunk_number:02d}/{total_chunks:02d}: intraday", flush=True)
            intraday = intraday_network(data, times, day_ahead)
            solve(intraday, solver)
            stage_flows["Intraday"].append(intraday.links_t.p0.copy())
        for stage, chunks in stage_flows.items():
            hourly = pd.concat(chunks).sort_index()
            if len(hourly) != 8_760:
                raise RuntimeError(f"{model} {stage} contains {len(hourly)} hours, expected 8,760")
            tables.append(summarize_flows(hourly, data["edges"], model, stage))
    return pd.concat(tables, ignore_index=True)


def add_ensemble_reference(flows: pd.DataFrame) -> pd.DataFrame:
    """Attach the four-model mean and each model's signed deviation from it."""
    reference = flows.groupby(["stage", "line"], as_index=False)["mean_flow_mw"].mean()
    reference = reference.rename(columns={"mean_flow_mw": "ensemble_mean_flow_mw"})
    output = flows.merge(reference, on=["stage", "line"], how="left", validate="many_to_one")
    output["model_deviation_mw"] = output.mean_flow_mw - output.ensemble_mean_flow_mw
    return output


def load_map_boundaries(path: Path) -> list[np.ndarray]:
    """Read map polylines stored in one local NetCDF boundary file."""
    if not path.exists():
        raise FileNotFoundError(f"Province boundary file not found: {path}")
    with netcdf_file(path, mode="r", mmap=False) as dataset:
        x = np.asarray(dataset.variables["x"][:], dtype=float)
        y = np.asarray(dataset.variables["y"][:], dtype=float)
        segments = np.asarray(dataset.variables["segments"][:], dtype=int)

    lines = []
    for start, count in segments:
        line = np.column_stack((x[start:start + count], y[start:start + count]))
        if len(line) >= 2 and np.isfinite(line).all():
            lines.append(line)
    if not lines:
        raise ValueError(f"No province boundary polylines found in {path}")
    return lines


def add_south_china_sea_inset(
    ax: plt.Axes,
    national: list[np.ndarray],
    provincial: list[np.ndarray],
) -> None:
    """Add an unobscured South China Sea islands inset to a China map panel."""
    inset = ax.inset_axes([0.825, 0.025, 0.155, 0.31], zorder=8)
    inset.set_facecolor("white")
    inset.add_collection(LineCollection(
        provincial, colors="#68717A", linewidths=0.34, alpha=0.95,
        capstyle="round", joinstyle="round", zorder=1,
    ))
    inset.add_collection(LineCollection(
        national, colors="#343A40", linewidths=0.56, alpha=1.0,
        capstyle="round", joinstyle="round", zorder=2,
    ))
    inset.set_xlim(105.0, 125.0)
    inset.set_ylim(3.0, 25.0)
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("#343A40")
        spine.set_linewidth(0.62)


def draw_lonlat_network(
    ax: plt.Axes,
    frame: pd.DataFrame,
    national_boundaries: list[np.ndarray],
    provincial_boundaries: list[np.ndarray],
    color: str,
    max_backbone: float,
    max_deviation: float,
    panel_label: str,
) -> None:
    """Draw the strongest shared links and model deviations on lon-lat axes."""
    backbone_cutoff = frame.ensemble_mean_flow_mw.abs().nlargest(min(14, len(frame))).min()
    deviation_cutoff = frame.model_deviation_mw.abs().nlargest(min(12, len(frame))).min()

    # Deliberately use outlines only: no land fill, raster layer, or thematic
    # shading. This keeps the province geography legible without competing
    # with the network-flow encoding.
    ax.add_collection(LineCollection(
        provincial_boundaries, colors="#68717A", linewidths=0.42, alpha=0.95,
        capstyle="round", joinstyle="round", zorder=0.4,
    ))
    ax.add_collection(LineCollection(
        national_boundaries, colors="#343A40", linewidths=0.72, alpha=1.0,
        capstyle="round", joinstyle="round", zorder=0.5,
    ))

    for row in frame.itertuples(index=False):
        if abs(row.ensemble_mean_flow_mw) < backbone_cutoff:
            continue
        x0, y0 = PROVINCE_COORDS[row.bus0]
        x1, y1 = PROVINCE_COORDS[row.bus1]
        width = 0.65 + 2.0 * abs(row.ensemble_mean_flow_mw) / max_backbone
        ax.plot([x0, x1], [y0, y1], color="#C9CDD1", lw=width,
                alpha=0.72, solid_capstyle="round", zorder=1)

    for row in frame.itertuples(index=False):
        deviation = row.model_deviation_mw
        if abs(deviation) < deviation_cutoff:
            continue
        start, end = (row.bus0, row.bus1) if deviation >= 0 else (row.bus1, row.bus0)
        x0, y0 = PROVINCE_COORDS[start]
        x1, y1 = PROVINCE_COORDS[end]
        scale = abs(deviation) / max_deviation
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "-|>", "color": color,
                "lw": 0.8 + 3.5 * scale, "alpha": 0.92,
                "mutation_scale": 7.0 + 5.0 * scale,
                "shrinkA": 0.0, "shrinkB": 0.0,
            },
            zorder=3,
        )

    coords = np.asarray(list(PROVINCE_COORDS.values()))
    ax.scatter(coords[:, 0], coords[:, 1], s=13, color="#333333",
               edgecolor="white", linewidth=0.45, zorder=4)
    ax.set_xlim(73, 135)
    ax.set_ylim(18, 54)
    ax.set_xticks([80, 90, 100, 110, 120, 130])
    ax.set_yticks([20, 30, 40, 50])
    ax.grid(color="#D9DDE2", lw=0.65, ls=":", zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="out", length=3.5, width=0.8)
    ax.text(0.015, 0.975, panel_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=19, fontweight="bold")
    add_south_china_sea_inset(ax, national_boundaries, provincial_boundaries)


def make_figure(flows: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12.5,
        "axes.labelsize": 13.5,
        "axes.linewidth": 0.75,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    scales = {
        stage: {
            "backbone": max(float(group.ensemble_mean_flow_mw.abs().max()), 1.0),
            "deviation": max(float(group.model_deviation_mw.abs().max()), 1.0),
        }
        for stage, group in flows.groupby("stage")
    }
    national_boundaries = load_map_boundaries(BOUNDARY_DIR / "cnmap.nc")
    provincial_boundaries = load_map_boundaries(BOUNDARY_DIR / "cnhimap.nc")
    fig, axes = plt.subplots(2, 4, figsize=(16.2, 6.35), sharex=True, sharey=True)
    for column, model in enumerate(MODELS):
        for row, stage in enumerate(("Day-ahead", "Intraday")):
            subset = flows.loc[(flows.model == model) & (flows.stage == stage)]
            draw_lonlat_network(
                axes[row, column], subset, national_boundaries,
                provincial_boundaries, MODEL_COLORS[model],
                scales[stage]["backbone"], scales[stage]["deviation"],
                chr(ord("a") + row * 4 + column),
            )
            if row == 0:
                axes[row, column].set_title(model, color=MODEL_COLORS[model],
                                            fontsize=18, fontweight="bold", pad=9)
            if row == 1:
                axes[row, column].set_xlabel(r"Longitude ($^\circ$E)")
            if column == 0:
                axes[row, column].set_ylabel(r"Latitude ($^\circ$N)")

    axes[0, 0].text(-0.22, 0.5, "Day-ahead", transform=axes[0, 0].transAxes,
                    rotation=90, ha="center", va="center", fontsize=17, fontweight="bold")
    axes[1, 0].text(-0.22, 0.5, "Intraday", transform=axes[1, 0].transAxes,
                    rotation=90, ha="center", va="center", fontsize=17, fontweight="bold")
    reference = [
        Line2D([0], [0], color="#C9CDD1", lw=2.5, label="Four-model mean flow"),
        Line2D([0], [0], color="#4A4A4A", lw=2.5, marker=">", markersize=7,
               label="Model deviation from mean"),
    ]
    fig.legend(handles=reference, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.52, 0.985), fontsize=15,
               columnspacing=2.2, handlelength=3.0, handletextpad=0.7)
    fig.subplots_adjust(left=0.072, right=0.995, top=0.865, bottom=0.075,
                        wspace=0.10, hspace=0.10)

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "03_transfer.png"
    pdf = output_dir / "03_transfer.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    fig.savefig(pdf, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)
    return png, pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="highs")
    parser.add_argument("--figure-dir", type=Path, default=FIGURES)
    parser.add_argument("--result-dir", type=Path, default=RESULTS)
    parser.add_argument("--plot-only", action="store_true", help="Rebuild the figure from the existing annual flow CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    flow_path = args.result_dir / "03_transfer_flows.csv"
    if args.plot_only:
        flows = pd.read_csv(flow_path)
        if "model_deviation_mw" not in flows:
            flows = add_ensemble_reference(flows)
    else:
        data = load_inputs()
        print("analysis period: 8,760 hours; leap day excluded")
        flows = add_ensemble_reference(run_dispatch(data, args.solver))
        flows.to_csv(flow_path, index=False, float_format="%.6f")
    png, pdf = make_figure(flows, args.figure_dir)
    print(f"saved flow data: {flow_path}")
    print(f"saved figure: {png}")
    print(f"saved figure: {pdf}")


if __name__ == "__main__":
    main()

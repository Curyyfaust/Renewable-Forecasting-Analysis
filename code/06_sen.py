"""Storage sensitivity in the present power system (Figure 06).

The experiment fixes the observed 2020 generation fleet, load, transmission
network, weather and four forecast products. Only battery power is varied from
0 to 20% of installed wind-plus-solar power, with a fixed four-hour duration.
Every forecast case is compared with a perfect-forecast case at the identical
storage level, isolating storage mitigation of forecast-error impacts from the
general operational value of storage.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results" / "06_sen"
FIG = ROOT / "figure"
OPERATION = ROOT / "PyPSA-china" / "main_dispatch.py"

MODELS = ("CNN", "LSTM", "Transformer", "DNN")
ALL_MODELS = ("Perfect",) + MODELS
SHORT = {"CNN": "CNN", "LSTM": "LSTM", "Transformer": "Transf.", "DNN": "DNN"}
MODEL_COLORS = {
    "CNN": "#2D9FD3",
    "LSTM": "#20A486",
    "Transformer": "#F29A38",
    "DNN": "#9857CC",
}
TECH_COLORS = {
    "wind": "#59C3BE",
    "solar": "#F6D66F",
    "hydro": "#91BCDA",
    "nuclear": "#C7B6DF",
    "coal": "#E98A65",
    "gas": "#F4B9AD",
}
BATTERY_COLOR = "#548DCF"
STORAGE_COLORS = {0.10: "#9ECAE1", 0.20: "#2868A6"}
SEASONS = ("Winter", "Spring", "Summer", "Autumn")
SEASON_DAYS = {"Winter": 90, "Spring": 92, "Summer": 92, "Autumn": 91}
SEASON_FOR_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

_WORKER_OPERATION = None
_WORKER_BASE = None
_WORKER_FACTORS = None


def load_operation():
    spec = importlib.util.spec_from_file_location("ssp15_dispatch", OPERATION)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _kmedoids(distance: np.ndarray, clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic partitioning-around-medoids for a small distance matrix."""
    if clusters < 1 or clusters > len(distance):
        raise ValueError("Invalid number of representative-day clusters")
    medoids = [int(np.argmin(distance.sum(axis=1)))]
    while len(medoids) < clusters:
        nearest = distance[:, medoids].min(axis=1)
        nearest[medoids] = -np.inf
        medoids.append(int(np.argmax(nearest)))
    medoids = np.asarray(medoids, dtype=int)
    for _ in range(50):
        labels = distance[:, medoids].argmin(axis=1)
        updated = medoids.copy()
        for cluster in range(clusters):
            members = np.flatnonzero(labels == cluster)
            if len(members):
                updated[cluster] = members[np.argmin(distance[np.ix_(members, members)].sum(axis=1))]
        if np.array_equal(updated, medoids):
            break
        medoids = updated
    labels = distance[:, medoids].argmin(axis=1)
    return medoids, labels


def choose_representative_days(data: dict, days_per_season: int = 3) -> pd.DataFrame:
    """Select diverse seasonal medoids using realised load and renewable state only."""
    hourly = pd.DataFrame(
        {
            "load": data["load"].sum(axis=1),
            "wind": data["wind"].sum(axis=1),
            "solar": data["solar"].sum(axis=1),
        }
    )
    hourly["date"] = hourly.index.floor("D")
    daily = hourly.groupby("date").agg(
        mean_load=("load", "mean"),
        peak_load=("load", "max"),
        load_sd=("load", "std"),
        mean_wind=("wind", "mean"),
        wind_sd=("wind", "std"),
        peak_wind=("wind", "max"),
        mean_solar=("solar", "mean"),
        peak_solar=("solar", "max"),
    )
    daily = daily[hourly.groupby("date").size().eq(24)]
    daily["season"] = daily.index.month.map(SEASON_FOR_MONTH)
    rows = []
    features = [
        "mean_load", "peak_load", "load_sd", "mean_wind", "wind_sd",
        "peak_wind", "mean_solar", "peak_solar",
    ]
    for season in SEASONS:
        block = daily[daily.season.eq(season)]
        scale = (block[features].quantile(0.75) - block[features].quantile(0.25)).replace(0, 1)
        values = ((block[features] - block[features].median()) / scale).to_numpy(dtype=float)
        distance = np.sqrt(((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2))
        medoids, labels = _kmedoids(distance, days_per_season)
        cluster_weights = np.bincount(labels, minlength=days_per_season)
        # The operational time series can omit one boundary day after time-zone
        # conversion. Preserve the intended 365-day seasonal calendar exactly.
        cluster_weights[np.argmax(cluster_weights)] += SEASON_DAYS[season] - cluster_weights.sum()
        for cluster, medoid in enumerate(medoids):
            date = block.index[medoid]
            members = np.flatnonzero(labels == cluster)
            rows.append(
                {
                    "season": season,
                    "cluster": cluster + 1,
                    "month": int(date.month),
                    "representative_date": date,
                    "weight_days": int(cluster_weights[cluster]),
                    "mean_cluster_distance": float(distance[members, medoid].mean()),
                    "max_cluster_distance": float(distance[members, medoid].max()),
                }
            )
    selected = pd.DataFrame(rows).sort_values(["season", "cluster"]).reset_index(drop=True)
    if len(selected) != 4 * days_per_season or selected.weight_days.sum() != 365:
        raise ValueError("Representative-day audit failed")
    return selected


def normalise_cached_weights(
    hourly: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Upgrade older caches so seasonal weights cover exactly 365 days."""
    selected = selected.copy()
    hourly = hourly.copy()
    for season, target in SEASON_DAYS.items():
        mask = selected.season.eq(season)
        difference = target - int(selected.loc[mask, "weight_days"].sum())
        if difference:
            largest = selected.loc[mask, "weight_days"].idxmax()
            selected.loc[largest, "weight_days"] += difference
    weight_map = selected.set_index("representative_date")["weight_days"]
    hourly["weight_days"] = hourly["representative_date"].map(weight_map)
    if hourly.weight_days.isna().any() or int(selected.weight_days.sum()) != 365:
        raise ValueError("Cached representative-day weights could not be reconciled")
    return hourly, selected


def state_definitions() -> pd.DataFrame:
    controlled = pd.read_csv(DATA / "ssp15_controlled_states.csv").set_index("year")
    rows = []
    current = controlled.loc[2020]
    for share in (0.00, 0.05, 0.10, 0.20):
        rows.append(
            {
                "scenario": f"2020_S{int(round(100 * share)):02d}",
                "year": 2020,
                "storage_share": share,
                "storage_hours": 4,
                "wind_tw": float(current.wind_tw),
                "solar_tw": float(current.solar_tw),
                "pathway": False,
            }
        )
    for year in (2030, 2050):
        state = controlled.loc[year]
        planned_share = float(state.storage_power_share)
        for share in (0.0, planned_share):
            rows.append(
                {
                    "scenario": f"{year}_S{int(round(100 * share)):02d}",
                    "year": year,
                    "storage_share": share,
                    "storage_hours": int(state.storage_hours),
                    "wind_tw": float(state.wind_tw),
                    "solar_tw": float(state.solar_tw),
                    "pathway": True,
                }
            )
    return pd.DataFrame(rows)


def scale_inputs(base: dict, state: pd.Series) -> dict:
    """Scale wind and solar consistently for controlled SSP1.5 capacity states."""
    data = {
        "times": base["times"],
        "provinces": base["provinces"],
        "load": base["load"].copy(),
        "wind": base["wind"].copy(),
        "solar": base["solar"].copy(),
        "scenarios": {model: values.copy() for model, values in base["scenarios"].items()},
        "caps": base["caps"].copy(),
        "edges": base["edges"],
    }
    for technology in ("wind", "solar"):
        target_mw = float(state[f"{technology}_tw"]) * 1e6
        factor = target_mw / float(data["caps"][technology].sum())
        data["caps"][technology] *= factor
        data[technology] *= factor
        if technology == "wind":
            for model in data["scenarios"]:
                data["scenarios"][model] *= factor
    return data


def add_storage(network, capacities: pd.DataFrame, share: float, hours: int) -> None:
    if share <= 0:
        return
    vre = capacities.wind + capacities.solar
    total_power = share * float(vre.sum())
    allocation = vre / vre.sum()
    eta = np.sqrt(0.90)
    for province, fraction in allocation.items():
        network.add(
            "StorageUnit",
            f"battery::{province}",
            bus=province,
            carrier="battery",
            p_nom=total_power * fraction,
            max_hours=hours,
            efficiency_store=eta,
            efficiency_dispatch=eta,
            cyclic_state_of_charge=True,
            marginal_cost=0.01,
        )


def solve_case(operation, base: dict, state: pd.Series, model: str, times: pd.DatetimeIndex, factors) -> pd.DataFrame:
    data = scale_inputs(base, state)
    provinces = data["provinces"]
    load = data["load"].loc[times]
    solar = data["solar"].loc[times]
    observed = data["wind"].loc[times]
    forecast = observed if model == "Perfect" else data["scenarios"][model].loc[times]

    da = operation.day_ahead(
        times, provinces, load, solar, forecast, data["caps"], data["edges"]
    )
    add_storage(da, data["caps"], state.storage_share, int(state.storage_hours))
    operation.solve(da)

    rt = operation.real_time(
        da, provinces, load, solar, observed, data["caps"], data["edges"]
    )
    add_storage(rt, data["caps"], state.storage_share, int(state.storage_hours))
    operation.solve(rt)

    zero = pd.Series(0.0, index=times)
    up = zero.copy()
    down = zero.copy()
    balancing_cost = zero.copy()
    co2 = zero.copy()
    fuel_cost = zero.copy()
    da_tech = {tech: zero.copy() for tech in ("wind", "solar", "hydro", "nuclear", "coal", "gas")}
    rt_tech = {tech: zero.copy() for tech in da_tech}
    for tech in da_tech:
        da_cols = da.generators.index[da.generators.carrier == tech]
        if len(da_cols):
            da_tech[tech] = da.generators_t.p.loc[times, da_cols].sum(axis=1)
    for province in provinces:
        for tech in operation.MC:
            old = f"{tech}::{province}"
            if old not in da.generators.index:
                continue
            u = rt.generators_t.p[f"up::{old}"]
            dn = rt.generators_t.p[f"down::{old}"]
            actual = (rt.generators_t.p[f"base::{old}"] + u + dn).clip(lower=0)
            up += u
            down += -dn
            balancing_cost += u * operation.UP[tech] + dn * (-operation.DOWN[tech])
            fuel_cost += actual * operation.MC[tech]
            co2 += actual * float(factors[tech])
            rt_tech[tech] += actual
    for tech in ("wind", "solar"):
        cols = rt.generators.index[rt.generators.carrier == tech]
        if len(cols):
            rt_tech[tech] = rt.generators_t.p.loc[times, cols].sum(axis=1)

    def storage_power(network):
        if network.storage_units.empty:
            return zero.copy()
        return network.storage_units_t.p.loc[times].sum(axis=1)

    da_battery = storage_power(da)
    rt_battery = storage_power(rt)
    available = observed.sum(axis=1) + solar.sum(axis=1)
    renewable_dispatch = rt_tech["wind"] + rt_tech["solar"]
    da_flow = da.links_t.p0.loc[times].abs().sum(axis=1) if len(da.links) else zero
    rt_flow = rt.links_t.p0.loc[times].abs().sum(axis=1) if len(rt.links) else zero
    out = pd.DataFrame(
        {
            "timestamp": times,
            "forecast_error_gw": (forecast.sum(axis=1) - observed.sum(axis=1)) / 1000,
            "load_gw": load.sum(axis=1) / 1000,
            "up_gw": up / 1000,
            "down_gw": down / 1000,
            "balancing_cost_million_cny": balancing_cost / 1e6,
            "total_cost_million_cny": (fuel_cost + balancing_cost) / 1e6,
            "direct_co2_mt": co2 / 1e6,
            "curtailment_gwh": (available - renewable_dispatch).clip(lower=0) / 1000,
            "da_battery_gw": da_battery / 1000,
            "rt_battery_gw": rt_battery / 1000,
            "link_adjustment_gwh": (rt_flow - da_flow).abs() / 1000,
            "link_net_change_gw": (rt_flow - da_flow) / 1000,
        }
    )
    for tech in da_tech:
        out[f"da_{tech}_gw"] = da_tech[tech].to_numpy() / 1000
        out[f"rt_{tech}_gw"] = rt_tech[tech].to_numpy() / 1000
    return out


def initialize_worker() -> None:
    global _WORKER_OPERATION, _WORKER_BASE, _WORKER_FACTORS
    logging.getLogger("pypsa").setLevel(logging.ERROR)
    logging.getLogger("linopy").setLevel(logging.ERROR)
    _WORKER_OPERATION = load_operation()
    _WORKER_BASE = _WORKER_OPERATION.inputs()
    _WORKER_FACTORS = pd.read_csv(
        _WORKER_OPERATION.FACTOR / "carrier_operational_co2_factors.csv"
    ).set_index("carrier").factor_t_co2_per_mwh


def solve_task(task) -> pd.DataFrame:
    state_dict, model, selected_records = task
    state = pd.Series(state_dict)
    frames = []
    for day in selected_records:
        start = pd.Timestamp(day["representative_date"])
        times = pd.date_range(start, periods=24, freq="h")
        result = solve_case(
            _WORKER_OPERATION, _WORKER_BASE, state, model, times, _WORKER_FACTORS
        )
        result.insert(0, "model", model)
        result.insert(0, "scenario", state.scenario)
        result["year"] = state.year
        result["storage_share"] = state.storage_share
        result["pathway"] = state.pathway
        result["month"] = day["month"]
        result["weight_days"] = day["weight_days"]
        result["representative_date"] = start
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def run_experiment(
    force: bool = False,
    workers: int = 4,
    days_per_season: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    total_days = 4 * days_per_season
    cache = OUT / f"06_storage_future_hourly_{total_days}d.csv.gz"
    selected_path = OUT / f"06_representative_days_{total_days}d.csv"
    if cache.exists() and selected_path.exists() and not force:
        hourly = pd.read_csv(cache, parse_dates=["timestamp", "representative_date"])
        selected = pd.read_csv(selected_path, parse_dates=["representative_date"])
        hourly, selected = normalise_cached_weights(hourly, selected)
        hourly.to_csv(cache, index=False, compression="gzip", float_format="%.7f")
        selected.to_csv(selected_path, index=False)
        hourly.to_csv(OUT / "06_storage_future_hourly.csv.gz", index=False, compression="gzip", float_format="%.7f")
        selected.to_csv(OUT / "06_representative_days.csv", index=False)
        states = state_definitions()
        return hourly, selected, states

    operation = load_operation()
    base = operation.inputs()
    states = state_definitions()
    current_cache = OUT / f"06_current_storage_hourly_{total_days}d.csv.gz"
    if current_cache.exists() and selected_path.exists() and not force:
        current = pd.read_csv(current_cache, parse_dates=["timestamp", "representative_date"])
        selected = pd.read_csv(selected_path, parse_dates=["representative_date"])
        current, selected = normalise_cached_weights(current, selected)
        current["scenario"] = current.scenario.str.replace("CURRENT_", "2020_", regex=False)
        current["year"] = 2020
        current["pathway"] = False
        states_to_solve = states[states.year.ne(2020)]
        frames = [current]
    else:
        selected = choose_representative_days(base, days_per_season=days_per_season)
        states_to_solve = states
        frames = []
    selected.to_csv(selected_path, index=False)
    selected_records = selected.to_dict("records")
    tasks = []
    for state in states_to_solve.itertuples(index=False):
        for model in ALL_MODELS:
            tasks.append((state._asdict(), model, selected_records))
    context = mp.get_context("spawn")
    with context.Pool(processes=max(1, workers), initializer=initialize_worker) as pool:
        for completed, result in enumerate(pool.imap_unordered(solve_task, tasks), start=1):
            frames.append(result)
            print(f"completed scenario-model blocks: {completed}/{len(tasks)}", flush=True)
    hourly = pd.concat(frames, ignore_index=True)
    hourly.to_csv(cache, index=False, compression="gzip", float_format="%.7f")
    hourly.to_csv(OUT / "06_storage_future_hourly.csv.gz", index=False, compression="gzip", float_format="%.7f")
    selected.to_csv(OUT / "06_representative_days.csv", index=False)
    return hourly, selected, states


def summarize(hourly: pd.DataFrame) -> pd.DataFrame:
    hourly = hourly.copy()
    hourly["conventional_adjustment_gw"] = hourly.up_gw + hourly.down_gw
    day = (
        hourly.groupby(
            [
                "scenario", "year", "storage_share", "pathway", "model",
                "representative_date", "month", "weight_days",
            ],
            as_index=False,
        )
        .agg(
            wind_mae_gw=("forecast_error_gw", lambda x: x.abs().mean()),
            battery_discharge_gwh=("rt_battery_gw", lambda x: x.clip(lower=0).sum()),
            conventional_adjustment_gwh=("conventional_adjustment_gw", "sum"),
            downward_adjustment_gwh=("down_gw", "sum"),
            link_adjustment_gwh=("link_adjustment_gwh", "sum"),
            curtailment_gwh=("curtailment_gwh", "sum"),
            balancing_cost_million_cny=("balancing_cost_million_cny", "sum"),
            total_cost_million_cny=("total_cost_million_cny", "sum"),
            direct_co2_mt=("direct_co2_mt", "sum"),
        )
    )
    mean_cols = ["wind_mae_gw"]
    sum_cols = [c for c in day.columns if c.endswith(("_gwh", "_cny", "_mt"))]
    records = []
    keys = ["scenario", "year", "storage_share", "pathway", "model"]
    for values, block in day.groupby(keys, sort=False):
        weights = block.weight_days.to_numpy()
        row = dict(zip(keys, values))
        for col in mean_cols:
            row[col] = np.average(block[col], weights=weights)
        for col in sum_cols:
            row[col] = float((block[col] * block.weight_days).sum())
        records.append(row)
    annual = pd.DataFrame(records)
    perfect_columns = {
        "balancing_cost_million_cny": "perfect_balancing_cost",
        "total_cost_million_cny": "perfect_cost",
        "direct_co2_mt": "perfect_co2",
        "battery_discharge_gwh": "perfect_battery",
        "conventional_adjustment_gwh": "perfect_conventional_adjustment",
        "downward_adjustment_gwh": "perfect_downward_adjustment",
        "link_adjustment_gwh": "perfect_link_adjustment",
        "curtailment_gwh": "perfect_curtailment",
    }
    perfect = annual[annual.model.eq("Perfect")][["scenario", *perfect_columns]].rename(columns=perfect_columns)
    annual = annual.merge(perfect, on="scenario", validate="many_to_one")
    annual["balancing_cost_penalty_bn_cny"] = (
        annual.balancing_cost_million_cny - annual.perfect_balancing_cost
    ) / 1000
    annual["cost_penalty_bn_cny"] = (annual.total_cost_million_cny - annual.perfect_cost) / 1000
    annual["co2_penalty_mt"] = annual.direct_co2_mt - annual.perfect_co2
    annual["battery_increment_twh"] = (annual.battery_discharge_gwh - annual.perfect_battery) / 1000
    for col in ["conventional_adjustment_gwh", "downward_adjustment_gwh", "link_adjustment_gwh", "curtailment_gwh"]:
        annual[col.replace("_gwh", "_twh")] = annual[col] / 1000
    annual["conventional_adjustment_penalty_twh"] = (
        annual.conventional_adjustment_gwh - annual.perfect_conventional_adjustment
    ) / 1000
    annual["downward_adjustment_penalty_twh"] = (
        annual.downward_adjustment_gwh - annual.perfect_downward_adjustment
    ) / 1000
    annual["link_adjustment_penalty_twh"] = (
        annual.link_adjustment_gwh - annual.perfect_link_adjustment
    ) / 1000
    annual["curtailment_penalty_twh"] = (
        annual.curtailment_gwh - annual.perfect_curtailment
    ) / 1000
    return annual


def _figure_policy_draft(hourly: pd.DataFrame, annual: pd.DataFrame, states: pd.DataFrame) -> None:
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 12.8, "axes.titlesize": 15.0,
        "axes.titleweight": "bold", "axes.labelsize": 13.5,
        "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
        "legend.fontsize": 13.5, "axes.linewidth": 1.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    operation = load_operation()
    base = operation.inputs()
    base_caps = base["caps"].sum()
    shares = np.sort(states.storage_share.unique())
    share_pct = shares * 100
    present = annual[annual.model.isin(MODELS)].copy()

    fig = plt.figure(figsize=(16.4, 10.8))
    outer = fig.add_gridspec(
        3, 1, height_ratios=[0.62, 1.72, 0.95], left=0.060, right=0.985,
        bottom=0.065, top=0.965, hspace=0.34,
    )

    def style(ax, axis="y"):
        ax.grid(axis=axis, color="#D9DDE2", lw=0.65, ls=":", zorder=0)
        ax.tick_params(direction="out", length=4, width=0.9)

    # Tables retained for auditability and downstream use.
    metrics_all = [
        ("cost_penalty_bn_cny", "Total-cost penalty", "bn CNY"),
        ("balancing_cost_penalty_bn_cny", "Balancing-cost penalty", "bn CNY"),
        ("co2_penalty_mt", r"Direct CO$_2$ penalty", "Mt"),
        ("conventional_adjustment_penalty_twh", "Conventional adjustment", "TWh"),
        ("downward_adjustment_penalty_twh", "Downward adjustment", "TWh"),
        ("link_adjustment_penalty_twh", "Link-flow response", "TWh"),
    ]
    response_rows = []
    for metric, _, unit in metrics_all:
        for model in MODELS:
            block = present[present.model.eq(model)].sort_values("storage_share")
            for share, value in zip(block.storage_share * 100, block[metric]):
                response_rows.append({"model": model, "storage_share_pct": share,
                                      "metric": metric, "value": value, "unit": unit})
    pd.DataFrame(response_rows).to_csv(
        OUT / "06_storage_response_curves.csv", index=False, float_format="%.7f"
    )

    effect_rows = []
    effect_metrics = [
        ("conventional_adjustment_penalty_twh", "TWh"),
        ("downward_adjustment_penalty_twh", "TWh"),
        ("battery_increment_twh", "TWh"),
        ("link_adjustment_penalty_twh", "TWh"),
        ("cost_penalty_bn_cny", "bn CNY"),
        ("co2_penalty_mt", "Mt"),
    ]
    for metric, unit in effect_metrics:
        for model in MODELS:
            block = present[present.model.eq(model)].set_index("storage_share")
            baseline = float(block.loc[0.0, metric])
            for share in (0.10, 0.20):
                effect_rows.append({
                    "metric": metric, "model": model,
                    "storage_share_pct": 100 * share,
                    "change_from_no_storage": float(block.loc[share, metric] - baseline),
                    "unit": unit,
                })
    pd.DataFrame(effect_rows).to_csv(
        OUT / "06_storage_effects.csv", index=False, float_format="%.7f"
    )

    policy_metrics = [
        "cost_penalty_bn_cny", "balancing_cost_penalty_bn_cny",
        "co2_penalty_mt", "conventional_adjustment_penalty_twh",
    ]
    policy_rows = []
    for model in MODELS:
        block = present[present.model.eq(model)].set_index("storage_share")
        for metric in policy_metrics:
            baseline = float(block.loc[0.0, metric])
            best = float(block[metric].min())
            available = baseline - best
            for share in (0.05, 0.10, 0.20):
                value = float(block.loc[share, metric])
                policy_rows.append({
                    "model": model, "storage_share_pct": 100 * share,
                    "metric": metric, "baseline": baseline, "value": value,
                    "reduction_pct": 100 * (baseline - value) / baseline,
                    "captured_pct": 100 * (baseline - value) / available if available > 0 else np.nan,
                })
    policy = pd.DataFrame(policy_rows)
    policy.to_csv(OUT / "06_policy_metrics.csv", index=False, float_format="%.7f")

    # a: process-led experimental design, following the reference figure's narrative flow.
    axa = fig.add_subplot(outer[0])
    axa.set_axis_off()
    axa.text(-0.025, 1.04, "a", transform=axa.transAxes, fontsize=22, fontweight="bold")
    axa.text(0.0, 1.04, "Experiment and decision pathway", transform=axa.transAxes,
             fontsize=16.5, fontweight="bold", va="bottom")
    vre_gw = float((base_caps.wind + base_caps.solar) / 1000)
    total_tw = float(base_caps.sum() / 1e6)
    boxes = [
        ("Observed system", f"2020 fleet: {total_tw:.2f} TW\nLoad, weather and grid fixed", "#DDF1EE"),
        ("Forecast cases", "CNN, LSTM, Transformer, DNN\nplus Perfect reference", "#E5F0F7"),
        ("Operational test", "Matched two-stage dispatch\n24 weighted representative days", "#EEE7F5"),
        ("Storage sweep", f"0, 5, 10 and 20% of VRE\n4 h; up to {0.2 * vre_gw:.1f} GW", "#FBE8D2"),
        ("Decision evidence", "Cost, CO$_2$, balancing,\nthermal and link responses", "#F7DFE5"),
    ]
    x_positions = np.linspace(0.005, 0.805, len(boxes))
    width, height, y0 = 0.175, 0.58, 0.19
    for index, (x0, (title, body, color)) in enumerate(zip(x_positions, boxes)):
        axa.add_patch(FancyBboxPatch(
            (x0, y0), width, height, transform=axa.transAxes,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=color, edgecolor="#7A7A7A", linewidth=0.9,
        ))
        axa.text(x0 + width / 2, y0 + 0.40, title, transform=axa.transAxes,
                 ha="center", va="center", fontsize=13.2, fontweight="bold")
        axa.text(x0 + width / 2, y0 + 0.19, body, transform=axa.transAxes,
                 ha="center", va="center", fontsize=10.6, linespacing=1.28)
        if index < len(boxes) - 1:
            axa.add_patch(FancyArrowPatch(
                (x0 + width + 0.008, y0 + height / 2),
                (x_positions[index + 1] - 0.008, y0 + height / 2),
                transform=axa.transAxes, arrowstyle="-|>", mutation_scale=14,
                linewidth=1.15, color="#555555",
            ))

    middle = outer[1].subgridspec(1, 2, width_ratios=[1.62, 0.78], wspace=0.25)

    # b: absolute forecast-error responses; the shaded band is the planning range.
    gsb = middle[0].subgridspec(2, 2, wspace=0.27, hspace=0.34)
    metrics_b = [
        ("cost_penalty_bn_cny", "Total-cost penalty", "bn CNY"),
        ("co2_penalty_mt", r"Direct CO$_2$ penalty", "Mt"),
        ("conventional_adjustment_penalty_twh", "Conventional adjustment", "TWh"),
        ("link_adjustment_penalty_twh", "Link-flow response", "TWh"),
    ]
    baxes = []
    for index, (metric, title, unit) in enumerate(metrics_b):
        ax = fig.add_subplot(gsb[index // 2, index % 2])
        baxes.append(ax)
        ax.axvspan(5, 10, color="#DCEFEA", alpha=0.72, zorder=0)
        for model in MODELS:
            block = present[present.model.eq(model)].sort_values("storage_share")
            xvalues = block.storage_share.to_numpy() * 100
            yvalues = block[metric].to_numpy()
            ax.plot(xvalues, yvalues, color=MODEL_COLORS[model], lw=2.0,
                    marker="o", ms=5.5, mec="white", mew=0.65, label=model)
        ax.axhline(0, color="#666666", lw=0.85, ls="--")
        ax.set_title(title, pad=4)
        ax.set_xticks(share_pct)
        if index >= 2:
            ax.set_xlabel("Battery power (% of wind + solar)")
        else:
            ax.set_xticklabels([])
        ax.set_ylabel(unit)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        style(ax)
    baxes[0].text(1.10, 1.27, "Storage response across forecast models",
                  transform=baxes[0].transAxes, ha="center", fontweight="bold", fontsize=15)
    baxes[0].text(-0.24, 1.27, "b", transform=baxes[0].transAxes, fontsize=22, fontweight="bold")
    baxes[0].text(0.38, 0.91, "planning range", transform=baxes[0].transAxes,
                  fontsize=9.5, color="#356C61", ha="center")
    baxes[0].legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(1.10, 1.09),
                    columnspacing=0.85, handletextpad=0.35, fontsize=13.5)

    # c: model robustness at the 10% / 4 h planning benchmark.
    axc = fig.add_subplot(middle[1])
    heat_metrics = policy_metrics
    heat_labels = ["Total\ncost", "Balancing\ncost", "CO$_2$", "Conventional\nadjustment"]
    heat = (
        policy[np.isclose(policy.storage_share_pct, 10.0)]
        .pivot(index="model", columns="metric", values="reduction_pct")
        .loc[list(MODELS), heat_metrics]
    )
    image = axc.imshow(heat.to_numpy(), cmap="YlGnBu", vmin=20, vmax=80, aspect="auto")
    axc.set_xticks(np.arange(len(heat_labels)), heat_labels)
    axc.set_yticks(np.arange(len(MODELS)), [SHORT[m] for m in MODELS])
    axc.set_title("Penalty reduction at 10% / 4 h", fontsize=14.5, pad=9, fontweight="bold")
    for row in range(len(MODELS)):
        for column in range(len(heat_metrics)):
            value = float(heat.iloc[row, column])
            axc.text(column, row, f"{value:.0f}%", ha="center", va="center",
                     fontsize=11.2, fontweight="bold",
                     color="white" if value > 58 else "#202020")
    axc.tick_params(length=0)
    for spine in axc.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=axc, orientation="horizontal", fraction=0.055, pad=0.13)
    colorbar.set_label("Reduction from no storage (%)", fontsize=11.5)
    colorbar.ax.tick_params(labelsize=10, length=2)
    axc.text(-0.18, 1.10, "c", transform=axc.transAxes, fontsize=22, fontweight="bold")

    lower = outer[2].subgridspec(1, 2, width_ratios=[1.05, 1.55], wspace=0.28)

    # d: how quickly each benefit saturates across the modeled storage range.
    axd = fig.add_subplot(lower[0])
    capture_labels = ["Total cost", "CO$_2$", "Balancing cost", "Conventional adjustment"]
    capture_metrics = [
        "cost_penalty_bn_cny", "co2_penalty_mt",
        "balancing_cost_penalty_bn_cny", "conventional_adjustment_penalty_twh",
    ]
    capture = policy.groupby(["storage_share_pct", "metric"])["captured_pct"].mean()
    ypos = np.arange(len(capture_metrics))[::-1]
    for y, metric in zip(ypos, capture_metrics):
        five = float(capture.loc[(5.0, metric)])
        ten = float(capture.loc[(10.0, metric)])
        axd.plot([five, ten], [y, y], color="#A8ADB2", lw=2.0, zorder=1)
        axd.scatter(five, y, s=70, color="#9ECAE1", edgecolor="white", lw=0.8, zorder=3)
        axd.scatter(ten, y, s=70, color="#2868A6", edgecolor="white", lw=0.8, zorder=3)
        axd.text(ten + 1.8, y, f"{ten:.0f}%", va="center", fontsize=10.5)
    axd.axvline(100, color="#777777", ls="--", lw=0.85)
    axd.set_xlim(50, 108)
    axd.set_yticks(ypos, capture_labels)
    axd.set_xlabel("Share of maximum modeled benefit captured (%)")
    axd.set_title("Diminishing returns", fontsize=15, fontweight="bold", pad=6)
    style(axd, "x")
    axd.legend(
        handles=[Line2D([0], [0], marker="o", lw=0, color=color, label=label, markersize=7.5)
                 for color, label in (("#9ECAE1", "5% / 4 h"), ("#2868A6", "10% / 4 h"))],
        frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.01),
        handletextpad=0.35, columnspacing=1.0, fontsize=11.5,
    )
    axd.text(-0.18, 1.18, "d", transform=axd.transAxes, fontsize=22, fontweight="bold")

    # e: concise, data-backed policy interpretation.
    axe = fig.add_subplot(lower[1])
    axe.set_axis_off()
    axe.text(-0.035, 1.07, "e", transform=axe.transAxes, fontsize=22, fontweight="bold")
    axe.text(0.0, 1.07, "Policy signals from the modeled range", transform=axe.transAxes,
             fontsize=15.5, fontweight="bold", va="bottom")
    cards = [
        (0.68, "1  Deploy the first 5% / 4 h broadly",
         "Captures 92% of total-cost and 93% of CO$_2$ benefit.", "#E1F2EE"),
        (0.37, "2  Use 10% / 4 h as the planning benchmark",
         "Captures about 100% of cost and CO$_2$ benefit and cuts\nconventional adjustment by 67% versus no storage.", "#DFECF7"),
        (0.06, "3  Expand beyond 10% selectively",
         "The 10-20% step cuts another 14.9 TWh of adjustment, but\naverage cost and CO$_2$ penalties flatten while link response rises 6.6 TWh.", "#FBE8D2"),
    ]
    for y0, title, body, color in cards:
        axe.add_patch(FancyBboxPatch(
            (0.0, y0), 0.99, 0.23, transform=axe.transAxes,
            boxstyle="round,pad=0.010,rounding_size=0.015",
            facecolor=color, edgecolor="#9A9A9A", linewidth=0.8,
        ))
        axe.text(0.025, y0 + 0.155, title, transform=axe.transAxes,
                 fontsize=12.2, fontweight="bold", va="center")
        axe.text(0.025, y0 + 0.065, body, transform=axe.transAxes,
                 fontsize=10.4, va="center", linespacing=1.25)
    axe.text(
        0.01, -0.055,
        "Storage complements forecasting: at 10% / 4 h, residual forecast penalties still average "
        "14.3 bn CNY and 40.1 Mt CO$_2$.",
        transform=axe.transAxes, fontsize=10.3, color="#444444", va="top",
    )
    axe.text(
        0.01, -0.145,
        "Values are means across four forecast models; each case is compared with Perfect at the same storage level.",
        transform=axe.transAxes, fontsize=9.5, color="#666666", va="top",
    )

    output = FIG / "06_sen"
    fig.savefig(output.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _figure_clear_draft(hourly: pd.DataFrame, annual: pd.DataFrame, states: pd.DataFrame) -> None:
    """Visual-only comparison of storage flexibility and benefits across years."""
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 12.5, "axes.titlesize": 15.0,
        "axes.titleweight": "bold", "axes.labelsize": 13.0,
        "xtick.labelsize": 11.0, "ytick.labelsize": 11.0,
        "legend.fontsize": 11.5, "axes.linewidth": 0.85,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    present = annual[annual.model.isin(MODELS)].copy()
    controlled = pd.read_csv(DATA / "ssp15_controlled_states.csv").set_index("year")
    years = (2020, 2030, 2050)
    planned_share = {2020: 0.0, 2030: 0.10, 2050: 0.20}
    flexibility_metrics = [
        ("balancing_cost_penalty_bn_cny", "Balancing cost"),
        ("conventional_adjustment_penalty_twh", "Conventional adjustment"),
        ("downward_adjustment_penalty_twh", "Downward adjustment"),
    ]
    benefit_metrics = [
        ("cost_penalty_bn_cny", "Total cost"),
        ("co2_penalty_mt", "Direct CO$_2$"),
    ]
    all_metrics = benefit_metrics + flexibility_metrics

    future_rows = []
    for year in (2030, 2050):
        target_share = planned_share[year]
        for model in MODELS:
            block = present[(present.year.eq(year)) & (present.model.eq(model))]
            baseline = block[np.isclose(block.storage_share, 0.0)].iloc[0]
            stored = block[np.isclose(block.storage_share, target_share)].iloc[0]
            for metric, label in all_metrics:
                base_value = float(baseline[metric])
                stored_value = float(stored[metric])
                future_rows.append({
                    "year": year,
                    "storage_share_pct": 100 * target_share,
                    "model": model,
                    "metric": metric,
                    "metric_label": label,
                    "no_storage": base_value,
                    "with_storage": stored_value,
                    "reduction_pct": 100 * (base_value - stored_value) / max(abs(base_value), 1e-9),
                })
    future_effects = pd.DataFrame(future_rows)
    future_effects.to_csv(
        OUT / "06_future_storage_effects.csv", index=False, float_format="%.7f"
    )

    response_rows = []
    for row in present.itertuples(index=False):
        for metric, label in all_metrics:
            response_rows.append({
                "scenario": row.scenario,
                "year": row.year,
                "storage_share_pct": 100 * row.storage_share,
                "model": row.model,
                "metric": metric,
                "metric_label": label,
                "value": getattr(row, metric),
            })
    pd.DataFrame(response_rows).to_csv(
        OUT / "06_storage_response_curves.csv", index=False, float_format="%.7f"
    )

    fig = plt.figure(figsize=(14.2, 11.4))
    outer = fig.add_gridspec(
        14, 12, left=0.075, right=0.985, top=0.965, bottom=0.075,
        wspace=0.80, hspace=0.95,
    )
    panel_letters = "abcd"

    def style(ax: plt.Axes, grid_axis: str = "y") -> None:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis=grid_axis, color="#DCE1E5", lw=0.65, ls=":", zorder=0)
        ax.tick_params(direction="out", length=3.5, width=0.8)

    def label_panel(ax: plt.Axes, index: int) -> None:
        ax.text(-0.14, 1.08, panel_letters[index], transform=ax.transAxes,
                fontsize=20, fontweight="bold", ha="left", va="top")

    # a | full-width scenario trajectory, matching the reference's top workflow band.
    ax = fig.add_subplot(outer[0:3, :])
    xpos = np.arange(len(years))
    wind = controlled.loc[list(years), "wind_tw"].to_numpy(dtype=float)
    solar = controlled.loc[list(years), "solar_tw"].to_numpy(dtype=float)
    battery = controlled.loc[list(years), "battery_power_tw"].to_numpy(dtype=float)
    ax.bar(xpos, wind, width=0.62, color=TECH_COLORS["wind"], label="Wind", zorder=2)
    ax.bar(xpos, solar, width=0.62, bottom=wind, color=TECH_COLORS["solar"], label="Solar", zorder=2)
    ax.set_xticks(xpos, years)
    ax.set_ylabel("Wind and solar capacity (TW)")
    ax.set_title("SSP1.5 capacity states", pad=7)
    style(ax)
    ax2 = ax.twinx()
    ax2.plot(xpos, battery, color=BATTERY_COLOR, marker="o", ms=7, lw=2.2,
             label="Battery power", zorder=4)
    ax2.set_ylabel("Battery power (TW)", color=BATTERY_COLOR)
    ax2.tick_params(axis="y", colors=BATTERY_COLOR)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, ncol=3,
              loc="upper left", columnspacing=0.9, handletextpad=0.35)
    arrow_y = 0.90 * float((wind + solar).max())
    for left, right in zip(xpos[:-1], xpos[1:]):
        ax.annotate(
            "", xy=(right - 0.38, arrow_y), xytext=(left + 0.38, arrow_y),
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.15),
            zorder=5,
        )
    label_panel(ax, 0)

    # b | dense dot-range response matrix: no bars and no radar geometry.
    current = present[present.year.eq(2020)]
    response_metrics = flexibility_metrics + benefit_metrics
    response_labels = [
        "Balancing cost", "Conventional adjustment", "Downward adjustment",
        "Total cost", "Direct CO$_2$",
    ]
    axb = fig.add_subplot(outer[4:9, :])
    storage_colors = {0.05: "#55B7DE", 0.10: "#EDA15D", 0.20: "#E15B6C"}
    shares = (0.05, 0.10, 0.20)
    share_offsets = {0.05: -0.22, 0.10: 0.0, 0.20: 0.22}
    model_markers = {"CNN": "o", "LSTM": "s", "Transformer": "D", "DNN": "^"}
    y_positions = np.arange(len(response_metrics), dtype=float)
    axb.axhspan(-0.48, 2.48, color="#EAF4F7", alpha=0.72, zorder=0)
    axb.axhspan(2.52, 4.48, color="#FFF1EA", alpha=0.72, zorder=0)

    for metric_index, (metric, _) in enumerate(response_metrics):
        means_for_path = []
        y_for_path = []
        for share in shares:
            reductions = []
            for model in MODELS:
                block = current[current.model.eq(model)].set_index("storage_share")
                base = float(block.loc[0.0, metric])
                stored = float(block.loc[share, metric])
                reductions.append(100 * (base - stored) / max(abs(base), 1e-9))
            reductions = np.asarray(reductions, dtype=float)
            y_value = metric_index + share_offsets[share]
            mean_value = float(reductions.mean())
            color = storage_colors[share]
            axb.hlines(y_value, reductions.min(), reductions.max(), color=color,
                       lw=2.0, alpha=0.72, zorder=2)
            axb.scatter([reductions.min(), reductions.max()], [y_value, y_value],
                        s=18, marker="|", color=color, linewidths=1.4, zorder=3)
            for model, value in zip(MODELS, reductions):
                axb.scatter(value, y_value, s=31, marker=model_markers[model],
                            color=color, alpha=0.62, edgecolor="white",
                            linewidth=0.55, zorder=4)
            axb.scatter(mean_value, y_value, s=78, marker="o", color=color,
                        edgecolor="#333333", linewidth=0.8, zorder=5)
            means_for_path.append(mean_value)
            y_for_path.append(y_value)
        axb.plot(means_for_path, y_for_path, color="#8D9398", lw=1.15,
                 alpha=0.70, zorder=1)

    axb.axvline(0, color="#777777", lw=0.85, ls="--", zorder=1)
    axb.set_xlim(0, 91)
    axb.set_ylim(len(response_metrics) - 0.48, -0.48)
    axb.set_yticks(y_positions, response_labels)
    axb.tick_params(axis="y", labelsize=12.0, pad=8, length=0)
    for label in axb.get_yticklabels():
        label.set_fontweight("semibold")
    axb.set_xlabel("Reduction from no storage (%)", labelpad=8)
    axb.set_xticks([0, 20, 40, 60, 80])
    axb.grid(axis="x", color="#CAD0D5", lw=0.75, ls=":", zorder=0)
    axb.spines[["right", "top"]].set_visible(False)
    axb.spines[["bottom", "left"]].set_color("#555555")
    share_handles = [
        matplotlib.lines.Line2D([0], [0], color=storage_colors[share], marker="o",
                                lw=2.0, ms=6.5, label=f"{int(share * 100)}% / 4 h")
        for share in shares
    ]
    first_legend = axb.legend(handles=share_handles, frameon=False, loc="upper left",
                              bbox_to_anchor=(0.015, 0.985), ncol=1,
                              borderaxespad=0.0, handlelength=1.8,
                              labelspacing=0.35)
    axb.add_artist(first_legend)
    model_handles = [
        matplotlib.lines.Line2D([0], [0], color="#666666", marker=model_markers[model],
                                lw=0, ms=5.2, label=SHORT[model])
        for model in MODELS
    ]
    axb.legend(handles=model_handles, frameon=False, loc="lower right",
               bbox_to_anchor=(0.985, 0.02), ncol=2, borderaxespad=0.0,
               columnspacing=0.9, handletextpad=0.3, fontsize=9.2)
    label_panel(axb, 1)

    # c | paired future summary using the same visual grammar as panel b.
    axc = fig.add_subplot(outer[10:14, 0:6])
    future_plot_metrics = benefit_metrics + flexibility_metrics
    future_plot_labels = [
        "Total cost", "Direct CO$_2$", "Balancing cost",
        "Conventional adjustment", "Downward adjustment",
    ]
    future_years = (2030, 2050)
    future_colors = {2030: "#8EC3DD", 2050: "#2868A6"}
    future_offsets = {2030: -0.14, 2050: 0.14}
    y_future = np.arange(len(future_plot_metrics), dtype=float)
    axc.axhspan(-0.45, 1.45, color="#FFF1EA", alpha=0.72, zorder=0)
    axc.axhspan(1.55, 4.45, color="#EAF4F7", alpha=0.72, zorder=0)
    for metric_index, (metric, _) in enumerate(future_plot_metrics):
        means, y_path = [], []
        for year in future_years:
            block = future_effects[
                future_effects.year.eq(year) & future_effects.metric.eq(metric)
            ]
            values_year = block.set_index("model").reindex(MODELS).reduction_pct.to_numpy(dtype=float)
            y_value = metric_index + future_offsets[year]
            mean_value = float(values_year.mean())
            color = future_colors[year]
            axc.hlines(y_value, values_year.min(), values_year.max(), color=color,
                       lw=2.1, alpha=0.78, zorder=2)
            axc.scatter([values_year.min(), values_year.max()], [y_value, y_value],
                        s=19, marker="|", color=color, linewidths=1.5, zorder=3)
            for model, value in zip(MODELS, values_year):
                axc.scatter(value, y_value, s=31, marker=model_markers[model],
                            color=color, alpha=0.68, edgecolor="white",
                            linewidth=0.5, zorder=4)
            axc.scatter(mean_value, y_value, s=78, marker="o", color=color,
                        edgecolor="#333333", linewidth=0.8, zorder=5)
            means.append(mean_value)
            y_path.append(y_value)
        axc.plot(means, y_path, color="#8D9398", lw=1.2, zorder=1)
    axc.set_xlim(0, 91)
    axc.set_ylim(len(future_plot_metrics) - 0.48, -0.48)
    axc.set_yticks(y_future, future_plot_labels)
    axc.tick_params(axis="y", labelsize=10.5, pad=6, length=0)
    axc.set_xticks([0, 20, 40, 60, 80])
    axc.set_xlabel("Reduction from no storage (%)")
    axc.grid(axis="x", color="#CAD0D5", lw=0.7, ls=":", zorder=0)
    axc.spines[["right", "top"]].set_visible(False)
    year_handles = [
        matplotlib.lines.Line2D([0], [0], color=future_colors[year], marker="o",
                                lw=2, ms=6.5,
                                label=f"{year}: {int(100 * planned_share[year])}% / 4 h")
        for year in future_years
    ]
    axc.legend(handles=year_handles, frameon=False, loc="lower left", ncol=1,
               borderaxespad=0.4, handlelength=1.8, labelspacing=0.35)
    label_panel(axc, 2)

    # d | model-level robustness heat map.
    ax = fig.add_subplot(outer[10:14, 6:12])
    heat_metrics = [metric for metric, _ in all_metrics]
    heat_labels = [
        "Total\ncost",
        "Direct\nCO$_2$",
        "Balancing\ncost",
        "Conventional\nadjustment",
        "Downward\nadjustment",
    ]
    row_index = pd.MultiIndex.from_product([(2030, 2050), MODELS], names=["year", "model"])
    heat = (
        future_effects.pivot(index=["year", "model"], columns="metric", values="reduction_pct")
        .reindex(row_index)[heat_metrics]
    )
    values = heat.to_numpy(dtype=float)
    vmin = min(-10.0, float(np.nanmin(values)))
    vmax = max(80.0, float(np.nanmax(values)))
    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    image = ax.imshow(values, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(heat_labels)), heat_labels)
    ax.tick_params(axis="x", labelsize=9, pad=5)
    ax.set_yticks(
        np.arange(len(row_index)),
        [f"{year}  {SHORT[model]}" for year, model in row_index],
    )
    ax.set_title("Future storage effects by model", pad=8)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(column, row, f"{value:.0f}", ha="center", va="center",
                    fontsize=8.5, color="white" if abs(norm(value) - 0.5) > 0.32 else "#222222")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.045, pad=0.18)
    colorbar.set_label("Reduction from no storage (%)", fontsize=11.5)
    colorbar.ax.tick_params(labelsize=9.5, length=2)
    label_panel(ax, 3)

    output = FIG / "06_sen"
    fig.savefig(output.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def figure(hourly: pd.DataFrame, annual: pd.DataFrame, states: pd.DataFrame) -> None:
    """Mechanism-rich, publication-style storage sensitivity synthesis."""
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 15.0,
        "axes.titlesize": 17.5, "axes.titleweight": "bold",
        "axes.labelsize": 15.5, "xtick.labelsize": 13.5,
        "ytick.labelsize": 13.5, "legend.fontsize": 12.2,
        "axes.linewidth": 0.85, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    present = annual[annual.model.isin(MODELS)].copy()
    current = present[present.year.eq(2020)].copy()
    controlled = pd.read_csv(DATA / "ssp15_controlled_states.csv").set_index("year")
    years = (2020, 2030, 2050)
    planned_share = {2020: 0.0, 2030: 0.10, 2050: 0.20}
    shares = (0.0, 0.05, 0.10, 0.20)
    share_pct = np.array([0, 5, 10, 20], dtype=float)
    response_metrics = [
        ("balancing_cost_penalty_bn_cny", "Balancing cost"),
        ("conventional_adjustment_penalty_twh", "Conventional adjustment"),
        ("downward_adjustment_penalty_twh", "Downward adjustment"),
        ("cost_penalty_bn_cny", "Total cost"),
        ("co2_penalty_mt", "Direct CO$_2$"),
    ]
    response_colors = ["#8767B7", "#249A8D", "#4B88C7", "#E07A5F", "#D84A6B"]
    future_metrics = response_metrics + [
        ("curtailment_twh", "Curtailment"),
        ("link_adjustment_twh", "Link-flow adjustment"),
    ]

    future_rows = []
    for year in (2030, 2050):
        target = planned_share[year]
        for model in MODELS:
            block = present[(present.year.eq(year)) & (present.model.eq(model))]
            baseline = block[np.isclose(block.storage_share, 0.0)].iloc[0]
            stored = block[np.isclose(block.storage_share, target)].iloc[0]
            for metric, label in future_metrics:
                base_value = float(baseline[metric])
                stored_value = float(stored[metric])
                future_rows.append({
                    "year": year, "model": model, "metric": metric,
                    "metric_label": label, "storage_share_pct": 100 * target,
                    "no_storage": base_value, "with_storage": stored_value,
                    "reduction_pct": 100 * (base_value - stored_value) / max(abs(base_value), 1e-9),
                })
    future_effects = pd.DataFrame(future_rows)
    future_effects.to_csv(OUT / "06_future_storage_effects.csv", index=False, float_format="%.7f")

    response_rows = []
    for row in present.itertuples(index=False):
        for metric, label in response_metrics:
            response_rows.append({
                "scenario": row.scenario, "year": row.year,
                "storage_share_pct": 100 * row.storage_share,
                "model": row.model, "metric": metric,
                "metric_label": label, "value": getattr(row, metric),
            })
    pd.DataFrame(response_rows).to_csv(
        OUT / "06_storage_response_curves.csv", index=False, float_format="%.7f"
    )

    fig = plt.figure(figsize=(18.3, 17.0))
    grid = fig.add_gridspec(
        13, 12, left=0.075, right=0.985, top=0.968, bottom=0.058,
        wspace=1.00, hspace=1.70,
    )

    def style(ax: plt.Axes, axis: str = "y") -> None:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis=axis, color="#D7DDE1", lw=0.65, ls=":", zorder=0)
        ax.tick_params(direction="out", length=3.2, width=0.8)

    def panel(ax: plt.Axes, letter: str, x: float = -0.13, y: float = 1.08) -> None:
        ax.text(x, y, letter, transform=ax.transAxes, fontsize=23,
                fontweight="bold", ha="left", va="top")

    # a | controlled SSP1.5 capacity states.
    axa = fig.add_subplot(grid[0:3, 0:5])
    xpos = np.arange(3)
    wind = controlled.loc[list(years), "wind_tw"].to_numpy(float)
    solar = controlled.loc[list(years), "solar_tw"].to_numpy(float)
    battery = controlled.loc[list(years), "battery_power_tw"].to_numpy(float)
    axa.bar(xpos, wind, width=0.58, color=TECH_COLORS["wind"], label="Wind", zorder=2)
    axa.bar(xpos, solar, width=0.58, bottom=wind, color=TECH_COLORS["solar"], label="Solar", zorder=2)
    for x, total in zip(xpos, wind + solar):
        axa.text(x, total + 0.08, f"{total:.2f}", ha="center", va="bottom", fontsize=13.0)
    axa.set_xticks(xpos, years)
    axa.set_ylabel("Wind and solar capacity (TW)")
    axa.set_title("SSP1.5 capacity states")
    style(axa)
    axa2 = axa.twinx()
    axa2.plot(xpos, battery, color=BATTERY_COLOR, marker="o", ms=6.5, lw=2.2,
              label="Battery power (TW)", zorder=4)
    for x, power in zip(xpos[1:], battery[1:]):
        axa2.text(x, power + 0.035, f"{power * 1000:.0f} GW\n{power * 4000:.0f} GWh",
                  ha="center", va="bottom", fontsize=11.8, color=BATTERY_COLOR)
    axa2.set_ylabel("")
    axa2.tick_params(axis="y", right=False, labelright=False)
    axa2.spines[["top", "left", "right"]].set_visible(False)
    h1, l1 = axa.get_legend_handles_labels()
    h2, l2 = axa2.get_legend_handles_labels()
    axa.legend(h1 + h2, l1 + l2, frameon=False, ncol=3, loc="upper left",
               columnspacing=0.8, handletextpad=0.35)
    panel(axa, "a")

    # b | complete present-day response curves with cross-model envelopes.
    axb = fig.add_subplot(grid[0:3, 5:12])
    for (metric, label), color in zip(response_metrics, response_colors):
        means, lows, highs = [], [], []
        for share in shares:
            reductions = []
            for model in MODELS:
                block = current[current.model.eq(model)].set_index("storage_share")
                base = float(block.loc[0.0, metric])
                value = float(block.loc[share, metric])
                reductions.append(100 * (base - value) / max(abs(base), 1e-9))
            means.append(np.mean(reductions))
            lows.append(np.min(reductions))
            highs.append(np.max(reductions))
        axb.fill_between(share_pct, lows, highs, color=color, alpha=0.10, linewidth=0)
        axb.plot(share_pct, means, color=color, marker="o", ms=4.8, lw=2.0, label=label)
    axb.axhline(0, color="#777777", lw=0.8, ls="--")
    axb.set_xticks(share_pct)
    axb.set_xlabel("Battery power (% of wind + solar)")
    axb.set_ylabel("Reduction from no storage (%)")
    axb.set_title("2020 storage response")
    axb.legend(frameon=False, ncol=3, loc="upper left", columnspacing=0.9)
    style(axb)
    panel(axb, "b", x=-0.04)

    # c-d | hourly balancing composition in the reference's paired 2 x 2 layout.
    hourly_plot = hourly.copy()
    hourly_plot["timestamp"] = pd.to_datetime(hourly_plot["timestamp"])
    hourly_plot["plot_date"] = hourly_plot["timestamp"].dt.normalize()

    def peak_date(scenario: str) -> pd.Timestamp:
        subset = hourly_plot[(hourly_plot.scenario.eq(scenario)) & hourly_plot.model.eq("CNN")]
        scores = subset.groupby("plot_date").forecast_error_gw.apply(lambda x: np.abs(x).sum())
        return pd.Timestamp(scores.idxmax())

    def day_block(scenario: str, date: pd.Timestamp | None = None) -> pd.DataFrame:
        subset = hourly_plot[(hourly_plot.scenario.eq(scenario)) & hourly_plot.model.eq("CNN")]
        if date is None or not subset.plot_date.eq(date).any():
            date = peak_date(scenario)
        return subset[subset.plot_date.eq(date)].sort_values("timestamp").copy()

    up_colors = ["#24677A", "#4EAAA7", "#F1AE55", "#9DD9E8"]
    down_colors = ["#31864A", "#65B96E", "#F1AE55", "#C9DE8A"]
    up_labels = ["Conventional", "Renewables", "Battery discharge", "Net imports"]
    down_labels = ["Conventional", "Renewables", "Battery charge", "Curtailment"]

    def balance_components(block: pd.DataFrame, direction: str) -> tuple[np.ndarray, list[np.ndarray]]:
        hours = block.timestamp.dt.hour.to_numpy(float)
        conventional = (block[[f"rt_{t}_gw" for t in ("hydro", "nuclear", "coal", "gas")]].to_numpy()
                        - block[[f"da_{t}_gw" for t in ("hydro", "nuclear", "coal", "gas")]].to_numpy())
        renewable = (block[["rt_wind_gw", "rt_solar_gw"]].to_numpy()
                     - block[["da_wind_gw", "da_solar_gw"]].to_numpy())
        if direction == "up":
            parts = [np.clip(conventional, 0, None).sum(axis=1),
                     np.clip(renewable, 0, None).sum(axis=1),
                     np.clip(block.rt_battery_gw.to_numpy(float), 0, None),
                     np.clip(block.link_net_change_gw.to_numpy(float), 0, None)]
        else:
            parts = [np.clip(-conventional, 0, None).sum(axis=1),
                     np.clip(-renewable, 0, None).sum(axis=1),
                     np.clip(-block.rt_battery_gw.to_numpy(float), 0, None),
                     np.clip(block.curtailment_gwh.to_numpy(float), 0, None)]
        return hours, parts

    def plot_balance(axp: plt.Axes, block: pd.DataFrame, direction: str,
                     title: str, show_y: bool, show_x: bool, legend: bool = False) -> None:
        hours, parts = balance_components(block, direction)
        labels = up_labels if direction == "up" else down_labels
        colors = up_colors if direction == "up" else down_colors
        axp.stackplot(hours, parts, labels=labels, colors=colors, alpha=0.96, linewidth=0)
        axp.set_xlim(0, 23)
        axp.set_xticks([0, 6, 12, 18])
        axp.set_title(title, fontsize=14.5, pad=5)
        if show_x:
            axp.set_xlabel("Hour", fontsize=13.5)
        else:
            axp.tick_params(labelbottom=False)
        if show_y:
            axp.set_ylabel("Upward balancing (GW)" if direction == "up" else "Downward balancing (GW)",
                           fontsize=13.5)
        else:
            axp.tick_params(labelleft=False)
        style(axp)
        if legend:
            axp.legend(frameon=False, fontsize=10.5, ncol=2, loc="upper left",
                       columnspacing=0.6, handlelength=1.3)

    cgrid = grid[3:7, 0:6].subgridspec(2, 2, wspace=0.18, hspace=0.26)
    date_2020 = peak_date("2020_S20")
    current_cases = [
        (day_block("2020_S00", date_2020), "No storage"),
        (day_block("2020_S20", date_2020), "20% / 4 h"),
    ]
    caxes = []
    for column, (block, title) in enumerate(current_cases):
        top = fig.add_subplot(cgrid[0, column], sharey=caxes[0] if column else None)
        bottom = fig.add_subplot(cgrid[1, column], sharex=top,
                                 sharey=caxes[1] if column else None)
        plot_balance(top, block, "up", title, show_y=column == 0, show_x=False,
                     legend=column == 0)
        plot_balance(bottom, block, "down", "", show_y=column == 0, show_x=True,
                     legend=column == 0)
        caxes.extend([top, bottom])
    panel(caxes[0], "c", x=-0.22, y=1.16)

    dgrid = grid[3:7, 6:12].subgridspec(2, 2, wspace=0.18, hspace=0.26)
    future_cases = [
        (day_block("2030_S10"), "2030: 10% / 4 h"),
        (day_block("2050_S20"), "2050: 20% / 4 h"),
    ]
    daxes = []
    for column, (block, title) in enumerate(future_cases):
        top = fig.add_subplot(dgrid[0, column], sharey=daxes[0] if column else None)
        bottom = fig.add_subplot(dgrid[1, column], sharex=top,
                                 sharey=daxes[1] if column else None)
        plot_balance(top, block, "up", title, show_y=column == 0, show_x=False,
                     legend=column == 0)
        plot_balance(bottom, block, "down", "", show_y=column == 0, show_x=True,
                     legend=column == 0)
        if column == 0:
            top.yaxis.set_label_coords(-0.14, 0.5)
            bottom.yaxis.set_label_coords(-0.14, 0.5)
        daxes.extend([top, bottom])
    panel(daxes[0], "d", x=-0.12, y=1.16)

    # e | four dumbbell summaries, matching the reference's bottom row.
    egrid = grid[7:10, :].subgridspec(1, 4, wspace=0.34)
    target_share = {2020: 0.20, 2030: 0.10, 2050: 0.20}
    dumbbell_metrics = [
        ("balancing_cost_penalty_bn_cny", "Balancing-cost penalty", "bn CNY"),
        ("cost_penalty_bn_cny", "Total-cost penalty", "bn CNY"),
        ("co2_penalty_mt", "Direct CO$_2$ penalty", "Mt"),
        ("conventional_adjustment_penalty_twh", "Conventional adjustment", "TWh"),
    ]
    comparison_years = (2020, 2030, 2050)
    eaxes = []
    for index, (metric, title, unit) in enumerate(dumbbell_metrics):
        axe = fig.add_subplot(egrid[0, index])
        for yy, year in enumerate(comparison_years):
            block = present[present.year.eq(year)]
            no_values = block[np.isclose(block.storage_share, 0.0)].set_index("model").reindex(MODELS)[metric].to_numpy(float)
            with_values = block[np.isclose(block.storage_share, target_share[year])].set_index("model").reindex(MODELS)[metric].to_numpy(float)
            no_mean, with_mean = float(no_values.mean()), float(with_values.mean())
            reduction = 100 * (no_mean - with_mean) / max(abs(no_mean), 1e-9)
            axe.hlines(yy, min(no_mean, with_mean), max(no_mean, with_mean), color="#AEB4B8", lw=1.8, zorder=1)
            axe.hlines(yy, no_values.min(), no_values.max(), color="#B9BEC1", lw=0.8, zorder=1)
            axe.hlines(yy, with_values.min(), with_values.max(), color="#2878B5", lw=0.8, zorder=1)
            axe.scatter(no_mean, yy, s=42, color="#AEB4B8", edgecolor="white", linewidth=0.5, zorder=3)
            axe.scatter(with_mean, yy, s=42, color="#1479B8", edgecolor="white", linewidth=0.5, zorder=3)
            midpoint = (no_mean + with_mean) / 2
            axe.text(midpoint, yy - 0.20, f"-{reduction:.0f}%", ha="center", va="bottom", fontsize=11.5)
        axe.set_yticks(np.arange(3), comparison_years)
        axe.set_ylim(2.45, -0.45)
        if index > 0:
            axe.tick_params(labelleft=False)
        else:
            axe.set_ylabel("Year")
        axe.set_xlabel(unit)
        axe.set_title(title, fontsize=14.5, pad=6)
        style(axe, axis="x")
        eaxes.append(axe)
    legend_handles = [
        Line2D([0], [0], marker="o", color="#AEB4B8", lw=0, ms=5.5, label="No storage"),
        Line2D([0], [0], marker="o", color="#1479B8", lw=0, ms=5.5, label="With storage"),
    ]
    eaxes[1].legend(handles=legend_handles, frameon=False, loc="upper right", fontsize=11.8)
    panel(eaxes[0], "e", x=-0.26, y=1.16)

    # f | system-level operating outcomes under the year-specific storage plan.
    fgrid = grid[10:13, :].subgridspec(1, 4, wspace=0.34)
    operating_metrics = [
        ("battery_discharge_gwh", "Battery discharge", "TWh", 0.001, "box"),
        ("curtailment_twh", "Curtailment", "TWh", 1.0, "bar"),
        ("link_adjustment_twh", "Link-flow adjustment", "TWh", 1.0, "box"),
        ("direct_co2_mt", "Direct CO$_2$ emissions", "Mt", 1.0, "bar"),
    ]
    no_color, storage_color = "#AEB4B8", "#1479B8"
    faxes = []
    for index, (metric, title, unit, scale, chart_type) in enumerate(operating_metrics):
        axf = fig.add_subplot(fgrid[0, index])
        no_means, with_means, no_err, with_err = [], [], [], []
        no_samples, with_samples = [], []
        for year in comparison_years:
            block = present[present.year.eq(year)]
            no_values = (block[np.isclose(block.storage_share, 0.0)]
                         .set_index("model").reindex(MODELS)[metric].to_numpy(float) * scale)
            with_values = (block[np.isclose(block.storage_share, target_share[year])]
                           .set_index("model").reindex(MODELS)[metric].to_numpy(float) * scale)
            no_mean, with_mean = float(no_values.mean()), float(with_values.mean())
            no_means.append(no_mean)
            with_means.append(with_mean)
            no_samples.append(no_values)
            with_samples.append(with_values)
            no_err.append([no_mean - no_values.min(), no_values.max() - no_mean])
            with_err.append([with_mean - with_values.min(), with_values.max() - with_mean])

        xx = np.arange(len(comparison_years), dtype=float)
        no_means = np.asarray(no_means)
        with_means = np.asarray(with_means)
        no_err = np.asarray(no_err).T
        with_err = np.asarray(with_err).T
        if chart_type == "bar":
            width = 0.34
            axf.bar(xx - width / 2, no_means, width, color=no_color, label="No storage", zorder=2)
            axf.bar(xx + width / 2, with_means, width, color=storage_color,
                    label="With storage", zorder=2)
            axf.errorbar(xx - width / 2, no_means, yerr=no_err, fmt="none",
                         ecolor="#747A7E", elinewidth=1.0, capsize=2.5, zorder=3)
            axf.errorbar(xx + width / 2, with_means, yerr=with_err, fmt="none",
                         ecolor="#0B5F91", elinewidth=1.0, capsize=2.5, zorder=3)
        else:
            box_width = 0.28
            no_positions = xx - 0.18
            with_positions = xx + 0.18
            box_style = dict(linewidth=1.15)
            median_style = dict(color="#263238", linewidth=1.5)
            no_box = axf.boxplot(
                no_samples, positions=no_positions, widths=box_width,
                patch_artist=True, showfliers=False, manage_ticks=False,
                boxprops=box_style, medianprops=median_style,
                whiskerprops=dict(color="#747A7E", linewidth=1.15),
                capprops=dict(color="#747A7E", linewidth=1.15),
            )
            with_box = axf.boxplot(
                with_samples, positions=with_positions, widths=box_width,
                patch_artist=True, showfliers=False, manage_ticks=False,
                boxprops=box_style, medianprops=median_style,
                whiskerprops=dict(color="#0B5F91", linewidth=1.15),
                capprops=dict(color="#0B5F91", linewidth=1.15),
            )
            for box in no_box["boxes"]:
                box.set(facecolor=no_color, edgecolor="#747A7E", alpha=0.88)
            for box in with_box["boxes"]:
                box.set(facecolor=storage_color, edgecolor="#0B5F91", alpha=0.92)
            jitter = np.linspace(-0.055, 0.055, len(MODELS))
            for x, no_values, with_values in zip(xx, no_samples, with_samples):
                axf.scatter(x - 0.18 + jitter, no_values, s=18, color="#747A7E",
                            edgecolor="white", linewidth=0.35, zorder=4)
                axf.scatter(x + 0.18 + jitter, with_values, s=18, color="#0B5F91",
                            edgecolor="white", linewidth=0.35, zorder=4)

        span = max(float(np.nanmax(np.r_[no_means, with_means])), 1e-9)
        for x, no_mean, with_mean in zip(xx, no_means, with_means):
            if abs(no_mean) > 1e-8:
                change = 100 * (no_mean - with_mean) / abs(no_mean)
                label = f"{-change:+.0f}%"
            else:
                label = f"{with_mean:.1f}"
            axf.text(x, max(no_mean, with_mean) + 0.055 * span, label,
                     ha="center", va="bottom", fontsize=11.5)
        axf.set_xticks(xx, comparison_years)
        axf.set_xlabel("Year")
        axf.set_ylabel(unit)
        axf.set_title(title, fontsize=14.5, pad=6)
        axf.margins(x=0.12, y=0.20)
        style(axf)
        faxes.append(axf)
    f_legend = [
        Patch(facecolor=no_color, edgecolor="#747A7E", label="No storage"),
        Patch(facecolor=storage_color, edgecolor="#0B5F91", label="With storage"),
    ]
    faxes[0].legend(handles=f_legend, frameon=False, loc="upper left", fontsize=11.8)
    panel(faxes[0], "f", x=-0.26, y=1.16)

    output = FIG / "06_sen"
    fig.savefig(output.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="ignore the representative-day cache")
    parser.add_argument("--workers", type=int, default=4, help="parallel scenario-model workers")
    parser.add_argument("--days-per-season", type=int, default=6, help="seasonal cluster medoids (default: 6, total: 24)")
    args = parser.parse_args()
    hourly, selected, states = run_experiment(
        force=args.force,
        workers=args.workers,
        days_per_season=args.days_per_season,
    )
    annual = summarize(hourly)
    total_days = 4 * args.days_per_season
    annual.to_csv(OUT / "06_storage_future_annual_metrics.csv", index=False, float_format="%.7f")
    annual[annual.year.eq(2020)].to_csv(
        OUT / "06_current_storage_annual_metrics.csv", index=False, float_format="%.7f"
    )
    figure(hourly, annual, states)
    tech = ("wind", "solar", "hydro", "nuclear", "coal", "gas")
    implied_shed = (
        hourly.load_gw
        - hourly[[f"rt_{name}_gw" for name in tech]].sum(axis=1)
        - hourly.rt_battery_gw
    )
    audit = pd.DataFrame(
        {
            "metric": [
                "scenario_model_blocks", "representative_days", "sample_hours",
                "weighted_days", "max_implied_load_shedding_gw",
            ],
            "value": [
                hourly[["scenario", "model"]].drop_duplicates().shape[0],
                len(selected),
                len(hourly),
                int(selected.weight_days.sum()),
                float(implied_shed.clip(lower=0).max()),
            ],
        }
    )
    audit.to_csv(OUT / "06_audit.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "storage sensitivity in current and controlled SSP1.5 renewable-capacity states",
        "weather_year": 2020,
        "temporal_design": f"{total_days} exogenous seasonal cluster medoids weighted to 365 days",
        "days_per_season": args.days_per_season,
        "selection_features": [
            "mean load", "peak load", "load SD", "mean wind", "wind SD",
            "peak wind", "mean solar", "peak solar",
        ],
        "system_boundary": "2020 load, weather shapes, conventional fleet and transmission fixed; wind and solar scaled to MESSAGE SSP1.5 capacity states",
        "controlled_variables": ["wind capacity", "solar capacity", "battery power"],
        "scenarios": states.to_dict("records"),
        "storage_power_shares_of_wind_plus_solar": {
            "2020": [0.0, 0.05, 0.10, 0.20],
            "2030": [0.0, 0.10],
            "2050": [0.0, 0.20],
        },
        "storage_duration_hours": 4,
        "storage_allocation": "provincial share proportional to installed provincial wind-plus-solar power",
        "matched_baseline": "each forecast case is compared with Perfect at the identical year and storage level",
        "penalty_definition": "Forecast minus Perfect within storage state",
        "storage_mitigation_definition": "penalty(storage) minus penalty(no storage); negative values indicate mitigation",
        "max_implied_load_shedding_gw": float(implied_shed.clip(lower=0).max()),
        "models": list(MODELS),
        "representative_dates": selected.assign(representative_date=selected.representative_date.astype(str)).to_dict("records"),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(annual[annual.model.isin(MODELS)].to_string(index=False))
    print(FIG / "06_sen.pdf")


if __name__ == "__main__":
    main()

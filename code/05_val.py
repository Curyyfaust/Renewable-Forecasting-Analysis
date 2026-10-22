"""Counterfactual validation of the spatiotemporal SHAP attribution.

The experiment decomposes provincial wind-forecast errors into temporal bias,
temporal variability, spatial bias and spatial variability.  Each component is
reduced to 50% and 0% while all other components are held fixed.  The modified
forecasts are then passed through the same two-stage PyPSA dispatch used by the
main experiment.  Resulting operational changes are compared with changes
predicted by the XGBoost surrogate underlying the SHAP analysis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figure"
FEATURE_FILE = RESULTS / "04_shap_features.csv"
DISPATCH_FILE = RESULTS / "pypsa_dispatch_2020_all_models.csv"
DISPATCH_MODULE = ROOT / "PyPSA-china" / "main_dispatch.py"

MODELS = ("CNN", "LSTM", "Transformer", "DNN")
SEASONS = ("Winter", "Spring", "Summer", "Autumn")
REGIONS = ("N", "E", "NW", "NE", "SW", "C", "S")
PROVINCE_REGION = {
    "Beijing": "N", "Tianjin": "N", "Hebei": "N", "Shanxi": "N", "InnerMongolia": "N",
    "Shanghai": "E", "Jiangsu": "E", "Zhejiang": "E", "Anhui": "E", "Fujian": "E",
    "Jiangxi": "E", "Shandong": "E", "Shaanxi": "NW", "Gansu": "NW", "Qinghai": "NW",
    "Ningxia": "NW", "Xinjiang": "NW", "Liaoning": "NE", "Jilin": "NE", "Heilongjiang": "NE",
    "Chongqing": "SW", "Sichuan": "SW", "Guizhou": "SW", "Yunnan": "SW", "Tibet": "SW",
    "Henan": "C", "Hubei": "C", "Hunan": "C", "Guangdong": "S", "Guangxi": "S", "Hainan": "S",
}

INTERVENTIONS = {
    "TB": ("Temporal bias", "#ED5675"),
    "TV": ("Temporal variability", "#F2A34A"),
    "SB": ("Spatial bias", "#55B7C9"),
    "SV": ("Spatial variability", "#67B96B"),
}
MODEL_COLORS = {"CNN": "#4C88CF", "LSTM": "#22B18A", "Transformer": "#F39C3D", "DNN": "#A45FD2"}
SHORT_LABELS = {"TB": "Temp. bias", "TV": "Temp. var.", "SB": "Spatial bias", "SV": "Spatial var."}
DYNAMIC_2022_FILE = RESULTS / "05_dynamic_linear_2022" / "2022_objective_specific_summary.csv"
DYNAMIC_2022_DAILY_FILE = RESULTS / "05_dynamic_linear_2022" / "2022_dispatch_daily.csv"
OUTCOME_COLORS = {
    "total_cost": "#ED6A4A",
    "balancing_cost": "#F2A34A",
    "co2": "#2F9ED3",
}

COUNTERFACTUAL_TARGETS = (
    ("total_cost", "total_operating_cost_million_cny", "total_cost_change_million_cny"),
    ("balancing_cost", "balancing_cost_million_cny", "balancing_cost_million_cny"),
    ("co2", "direct_co2_mt", "co2_change_mt"),
)


def load_dispatch_module():
    spec = importlib.util.spec_from_file_location("counterfactual_dispatch", DISPATCH_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load dispatch module: {DISPATCH_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def new_model(seed: int = 42) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=450, max_depth=3, learning_rate=0.025, min_child_weight=5,
        subsample=0.82, colsample_bytree=0.82, reg_alpha=0.04, reg_lambda=1.2,
        objective="reg:squarederror", random_state=seed, n_jobs=8,
    )


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "model", "day_id", "date", "season", "total_cost_change_million_cny",
        "balancing_cost_million_cny", "co2_change_mt",
    }
    return [column for column in frame.columns if column not in excluded]


def longest_sign_run(values: np.ndarray) -> int:
    signs = np.sign(values).copy()
    signs[signs == 0] = 1
    changes = np.flatnonzero(np.r_[True, signs[1:] != signs[:-1], True])
    return int(np.diff(changes).max())


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def extract_features(
    model: str,
    day_ids: list[int],
    prediction: np.ndarray,
    truth: np.ndarray,
    provinces: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    operation_dates = pd.date_range("2020-01-01", periods=365, freq="D")
    error = prediction - truth
    for local_index, day_id in enumerate(day_ids):
        e = error[local_index]
        pred = prediction[local_index]
        actual = truth[local_index]
        national = e.sum(axis=1) / 1_000.0
        predicted_national = pred.sum(axis=1) / 1_000.0
        actual_national = actual.sum(axis=1) / 1_000.0
        province_bias = e.mean(axis=0) / 1_000.0
        province_abs_energy = np.abs(e).sum(axis=0)
        shares = province_abs_energy / max(province_abs_energy.sum(), 1e-12)
        date = operation_dates[day_id]
        row: dict[str, float | int | str] = {
            "model": model,
            "day_id": day_id,
            "date": date,
            "season": season_from_month(date.month),
            "underforecast_bias_gw": float(np.clip(-national, 0, None).mean()),
            "overforecast_bias_gw": float(np.clip(national, 0, None).mean()),
            "maximum_hourly_error_gw": float(np.abs(national).max()),
            "temporal_error_sd_gw": float(national.std(ddof=0)),
            "ramp_error_gw": float(np.sqrt(np.mean(np.diff(predicted_national - actual_national) ** 2))),
            "error_persistence_h": longest_sign_run(national),
            "provincial_abs_bias_gw": float(np.abs(province_bias).mean()),
            "spatial_error_sd_gw": float(province_bias.std(ddof=0)),
            "spatial_p90_gw": float(np.quantile(np.abs(province_bias), 0.90)),
            "spatial_concentration": float(np.square(shares).sum()),
        }
        for region in REGIONS:
            indices = [i for i, province in enumerate(provinces) if PROVINCE_REGION[province] == region]
            regional_hourly = e[:, indices].sum(axis=1) / 1_000.0
            regional_bias = e[:, indices].mean(axis=0) / 1_000.0
            row[f"region_{region}_under_gw"] = float(np.clip(-regional_hourly, 0, None).mean())
            row[f"region_{region}_over_gw"] = float(np.clip(regional_hourly, 0, None).mean())
            row[f"region_{region}_dispersion_gw"] = float(regional_bias.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def select_representative_days(features: pd.DataFrame, days_per_season: int) -> pd.DataFrame:
    severity = (
        features.groupby(["day_id", "date", "season"], as_index=False)
        .maximum_hourly_error_gw.mean()
    )
    selected = []
    for season in SEASONS:
        subset = severity.loc[severity.season == season].sort_values("maximum_hourly_error_gw").reset_index(drop=True)
        indices = np.unique(np.rint(np.linspace(0, len(subset) - 1, days_per_season)).astype(int))
        if len(indices) < days_per_season:
            remaining = [index for index in range(len(subset)) if index not in set(indices)]
            indices = np.sort(np.r_[indices, remaining[: days_per_season - len(indices)]])
        chosen = subset.iloc[indices].copy()
        chosen["severity_quantile"] = np.linspace(0, 1, len(chosen))
        selected.append(chosen)
    return pd.concat(selected, ignore_index=True).sort_values("day_id").reset_index(drop=True)


def decompose_error(prediction: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    error = prediction - truth
    national = error.sum(axis=2)
    totals = truth.sum(axis=2, keepdims=True)
    weights = np.divide(truth, totals, out=np.full_like(truth, 1.0 / truth.shape[2]), where=totals > 1e-9)
    daily_bias = national.mean(axis=1, keepdims=True)
    temporal_variability = national - daily_bias
    spatial = error - weights * national[:, :, None]
    spatial_bias = spatial.mean(axis=1, keepdims=True)
    spatial_variability = spatial - spatial_bias
    return {
        "weights": weights,
        "TB": weights * daily_bias[:, :, None],
        "TV": weights * temporal_variability[:, :, None],
        "SB": np.broadcast_to(spatial_bias, error.shape).copy(),
        "SV": spatial_variability,
    }


def counterfactual_prediction(
    truth: np.ndarray,
    components: dict[str, np.ndarray],
    intervention: str,
    alpha: float,
    capacity: np.ndarray,
) -> tuple[np.ndarray, float]:
    modified = sum(alpha * values if key == intervention else values for key, values in components.items() if key != "weights")
    unrestricted = truth + modified
    clipped = np.clip(unrestricted, 0, capacity[None, None, :])
    clipped_share = float(np.mean(np.abs(clipped - unrestricted) > 1e-8))
    return clipped, clipped_share


def raw_daily_outcomes(dispatch: pd.DataFrame, selected_days: list[int]) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        block = dispatch.loc[dispatch.model == model].sort_values("timestamp_beijing").reset_index(drop=True)
        for day_id in selected_days:
            day = block.iloc[day_id * 24:(day_id + 1) * 24]
            rows.append({
                "model": model, "day_id": day_id, "scenario": "Raw",
                "intervention": "Raw", "alpha": 1.0,
                "total_operating_cost_million_cny": float(day.total_operating_cost_cny.sum() / 1e6),
                "balancing_cost_million_cny": float(day.balancing_cost_cny.sum() / 1e6),
                "direct_co2_mt": float(day.direct_co2_t.sum() / 1e6),
                "clipped_share": 0.0,
            })
    return pd.DataFrame(rows)


def aggregate_counterfactual(hourly: pd.DataFrame, model: str, day_ids: list[int], scenario: str, intervention: str, alpha: float, clipped_share: float) -> pd.DataFrame:
    rows = []
    for local_index, day_id in enumerate(day_ids):
        day = hourly.iloc[local_index * 24:(local_index + 1) * 24]
        rows.append({
            "model": model, "day_id": day_id, "scenario": scenario,
            "intervention": intervention, "alpha": alpha,
            "total_operating_cost_million_cny": float(day.total_operating_cost_cny.sum() / 1e6),
            "balancing_cost_million_cny": float(day.balancing_cost_cny.sum() / 1e6),
            "direct_co2_mt": float(day.direct_co2_t.sum() / 1e6),
            "clipped_share": clipped_share,
        })
    return pd.DataFrame(rows)


def fit_surrogates(features: pd.DataFrame) -> tuple[list[str], dict[str, XGBRegressor]]:
    columns = feature_columns(features)
    models = {}
    for key, _, target in COUNTERFACTUAL_TARGETS:
        model = new_model()
        model.fit(features[columns], features[target])
        models[key] = model
    return columns, models


def date_clustered_mean_ci(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    multiplier: float = 1.0,
    repeats: int = 1000,
) -> tuple[float, float, float]:
    """Bootstrap dates within seasons while retaining all rows from each date."""
    frame = frame[["season", "day_id", "actual_delta"]].copy()
    frame["actual_delta"] = frame["actual_delta"].astype(float) * multiplier
    seasons = tuple(frame["season"].drop_duplicates())
    clusters = {
        season: {
            day_id: group["actual_delta"].to_numpy(dtype=float)
            for day_id, group in frame.loc[frame.season == season].groupby("day_id", sort=False)
        }
        for season in seasons
    }
    means = np.empty(repeats)
    for index in range(repeats):
        sampled_values = []
        for season in seasons:
            day_ids = np.asarray(list(clusters[season]))
            sampled_days = rng.choice(day_ids, size=len(day_ids), replace=True)
            sampled_values.extend(clusters[season][day_id] for day_id in sampled_days)
        means[index] = np.concatenate(sampled_values).mean()
    values = frame["actual_delta"].to_numpy(dtype=float)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def moving_block_annual_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    block_days: int = 7,
    repeats: int = 5000,
) -> tuple[float, float]:
    """Return a paired moving-block bootstrap CI for an annual sum.

    Circular seven-day blocks retain short-range serial dependence while each
    bootstrap replicate contains the same number of operational days.
    """
    values = np.asarray(values, dtype=float)
    sample_count = len(values)
    blocks_per_draw = int(np.ceil(sample_count / block_days))
    draws = np.empty(repeats, dtype=float)
    offsets = np.arange(block_days)
    for index in range(repeats):
        starts = rng.integers(0, sample_count, size=blocks_per_draw)
        indices = ((starts[:, None] + offsets[None, :]) % sample_count).ravel()[:sample_count]
        draws[index] = values[indices].sum()
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def dynamic_reduction_uncertainty(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build paired daily reductions and annual 95% confidence intervals."""
    specifications = (
        ("Total operating cost", "cost", "total_operating_cost_million_cny", 1 / 1000, "bn CNY"),
        ("Direct CO2", "co2", "direct_co2_mt", 1.0, "Mt"),
    )
    daily_rows = []
    summary_rows = []
    for model_index, model in enumerate(MODELS):
        model_data = daily.loc[daily.model == model].copy()
        for outcome_index, (objective, variant, column, scale, unit) in enumerate(specifications):
            paired = model_data.pivot(index="date", columns="variant", values=column).sort_index()
            reductions = (paired["Raw"] - paired[variant]) * scale
            reductions = reductions.dropna()

            # Beijing timestamps split the first/last operational day across
            # calendar-year boundaries. Combine those two partial days so the
            # uncertainty sample represents 365 complete operational days.
            if len(reductions) == 366:
                boundary = float(reductions.iloc[0] + reductions.iloc[-1])
                reductions = pd.concat([
                    pd.Series([boundary], index=["boundary operational day"]),
                    reductions.iloc[1:-1],
                ])

            for day_index, (date, value) in enumerate(reductions.items(), start=1):
                daily_rows.append({
                    "model": model,
                    "objective": objective,
                    "operational_day": day_index,
                    "source_date": date,
                    "reduction": float(value),
                    "unit": unit,
                })

            rng = np.random.default_rng(202205 + 10 * model_index + outcome_index)
            low, high = moving_block_annual_ci(reductions.to_numpy(), rng)
            summary_rows.append({
                "model": model,
                "objective": objective,
                "annual_reduction": float(reductions.sum()),
                "ci95_low": low,
                "ci95_high": high,
                "block_days": 7,
                "bootstrap_repeats": 5000,
                "paired_operational_days": len(reductions),
                "unit": unit,
            })
    return pd.DataFrame(daily_rows), pd.DataFrame(summary_rows)


def build_validation(
    outcomes: pd.DataFrame,
    cf_features: pd.DataFrame,
    raw_features: pd.DataFrame,
    columns: list[str],
    surrogate_models: dict[str, XGBRegressor],
) -> pd.DataFrame:
    raw_outcome = outcomes.loc[outcomes.scenario == "Raw"].set_index(["model", "day_id"])
    raw_lookup = raw_features.set_index(["model", "day_id"])
    rows = []
    for row in cf_features.itertuples(index=False):
        key = (row.model, row.day_id)
        outcome = outcomes.loc[
            (outcomes.model == row.model) & (outcomes.day_id == row.day_id) & (outcomes.scenario == row.scenario)
        ].iloc[0]
        x_cf = pd.DataFrame([{column: getattr(row, column) for column in columns}])
        x_raw = raw_lookup.loc[[key], columns]
        for target, outcome_column, _ in COUNTERFACTUAL_TARGETS:
            actual_delta = float(outcome[outcome_column] - raw_outcome.loc[key, outcome_column])
            predicted_delta = float(surrogate_models[target].predict(x_cf)[0] - surrogate_models[target].predict(x_raw)[0])
            rows.append({
                "model": row.model, "day_id": row.day_id, "season": row.season,
                "scenario": row.scenario, "intervention": row.intervention, "alpha": row.alpha,
                "target": target, "actual_delta": actual_delta, "predicted_delta": predicted_delta,
            })
    return pd.DataFrame(rows)


def validation_summary(validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, _, _ in COUNTERFACTUAL_TARGETS:
        subset = validation.loc[validation.target == target]
        denominator = float(np.dot(subset.predicted_delta, subset.predicted_delta))
        slope = float(np.dot(subset.predicted_delta, subset.actual_delta) / denominator) if denominator > 0 else np.nan
        rows.append({
            "target": target,
            "samples": len(subset),
            "counterfactual_r2": float(r2_score(subset.actual_delta, subset.predicted_delta)),
            "calibration_slope": slope,
            "mae": float(np.mean(np.abs(subset.actual_delta - subset.predicted_delta))),
            "spearman_r": float(spearmanr(subset.actual_delta, subset.predicted_delta).statistic),
            "direction_agreement_pct": float(100 * np.mean(np.sign(subset.actual_delta) == np.sign(subset.predicted_delta))),
        })
    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.9)
    ax.tick_params(width=0.85, length=4)
    if grid:
        ax.grid(True, color="#E7EBEF", linewidth=0.7, zorder=0)


def panel_label(ax: plt.Axes, label: str, x: float = -0.18, y: float = 1.12) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=25, fontweight="bold", ha="left", va="top")


def plot_design(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.88, "Forecast-error intervention", fontsize=16, fontweight="bold", ha="left")
    starts = [0.03, 0.27, 0.51, 0.75]
    for (code, (label, color)), x in zip(INTERVENTIONS.items(), starts):
        ax.add_patch(plt.Rectangle((x, 0.58), 0.20, 0.16, facecolor=color, edgecolor="none", alpha=0.92))
        ax.text(x + 0.10, 0.66, label.replace(" ", "\n"), color="white", fontsize=11.5,
                fontweight="bold", ha="center", va="center", linespacing=0.92)
    ax.annotate("", xy=(0.50, 0.42), xytext=(0.50, 0.56), arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#555555"})
    ax.text(0.50, 0.36, "50% reduction  /  complete removal", fontsize=13, ha="center", va="center")
    ax.annotate("", xy=(0.50, 0.20), xytext=(0.50, 0.31), arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#555555"})
    ax.text(0.50, 0.12, "Two-stage PyPSA redispatch", fontsize=14, fontweight="bold", ha="center", va="center")


def plot_calibration(ax: plt.Axes, validation: pd.DataFrame, target: str, title: str, summary: pd.DataFrame) -> None:
    subset = validation.loc[validation.target == target]
    multiplier = 1000 if target == "co2" else 1
    for code, (label, color) in INTERVENTIONS.items():
        block = subset.loc[subset.intervention == code]
        ax.scatter(block.predicted_delta * multiplier, block.actual_delta * multiplier, s=18, color=color,
                   alpha=0.48, edgecolors="none", rasterized=True, label=label, zorder=2)
    values = np.r_[subset.predicted_delta.to_numpy(), subset.actual_delta.to_numpy()] * multiplier
    lower, upper = np.quantile(values, [0.01, 0.99])
    margin = 0.08 * max(upper - lower, 1e-9)
    lower, upper = lower - margin, upper + margin
    ax.plot([lower, upper], [lower, upper], color="#333333", lw=1.2, ls="--", zorder=1)
    metric = summary.loc[summary.target == target].iloc[0]
    ax.text(0.04, 0.95, f"$R^2$ = {metric.counterfactual_r2:.2f}\nSign = {metric.direction_agreement_pct:.0f}%",
            transform=ax.transAxes, ha="left", va="top", fontsize=11.5)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("SHAP-surrogate $\Delta$")
    ax.set_ylabel("PyPSA $\Delta$")
    style_axis(ax)


def plot_dose(ax: plt.Axes, validation: pd.DataFrame, target: str, title: str, ylabel: str) -> None:
    subset = validation.loc[validation.target == target].copy()
    multiplier = 1000 if target == "co2" else 1
    rng = np.random.default_rng(42 if target == "cost" else 43)
    for code, (label, color) in INTERVENTIONS.items():
        x_values = [1.0, 0.5, 0.0]
        means, lows, highs = [0.0], [0.0], [0.0]
        for alpha in (0.5, 0.0):
            values = subset.loc[(subset.intervention == code) & np.isclose(subset.alpha, alpha)]
            mean, low, high = date_clustered_mean_ci(values, rng, multiplier)
            means.append(mean); lows.append(low); highs.append(high)
        ax.plot(x_values, means, marker="o", color=color, lw=2.2, ms=6, label=label, zorder=3)
        ax.fill_between(x_values, lows, highs, color=color, alpha=0.13, linewidth=0)
    ax.axhline(0, color="#666666", lw=0.9, ls="--")
    ax.set_xticks([0, 0.5, 1.0])
    ax.invert_xaxis()
    ax.set_xlabel("Remaining error component")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    style_axis(ax)


def plot_model_effect(ax_cost: plt.Axes, ax_co2: plt.Axes, validation: pd.DataFrame) -> None:
    rng = np.random.default_rng(44)
    positions = np.arange(len(MODELS))[::-1]
    offsets = np.linspace(-0.18, 0.18, len(INTERVENTIONS))
    for ax, target, multiplier, xlabel in (
        (ax_cost, "cost", 1, "$\Delta$ cost (M CNY day$^{-1}$)"),
        (ax_co2, "co2", 1000, "$\Delta$ CO$_2$ (kt day$^{-1}$)"),
    ):
        zero = validation.loc[(validation.target == target) & np.isclose(validation.alpha, 0.0)]
        for offset, (code, (label, color)) in zip(offsets, INTERVENTIONS.items()):
            for position, model in zip(positions, MODELS):
                values = zero.loc[(zero.model == model) & (zero.intervention == code)]
                mean, low, high = date_clustered_mean_ci(values, rng, multiplier)
                ax.errorbar(mean, position + offset, xerr=[[mean - low], [high - mean]], fmt="o", color=color,
                            ms=5.5, capsize=2, lw=1.2, zorder=3)
        ax.axvline(0, color="#666666", lw=0.9, ls="--")
        ax.set_yticks(positions, MODELS if ax is ax_cost else [])
        ax.set_xlabel(xlabel)
        style_axis(ax)
    ax_cost.set_title("Balancing cost", fontweight="bold")
    ax_co2.set_title("CO$_2$ emissions", fontweight="bold")


def plot_effect_bars(ax: plt.Axes, validation: pd.DataFrame, target: str, title: str, xlabel: str) -> None:
    subset = validation.loc[validation.target == target]
    multiplier = 1000 if target == "co2" else 1
    rng = np.random.default_rng(51 if target == "cost" else 52)
    positions = np.arange(len(INTERVENTIONS))[::-1]
    height = 0.27
    for position, (code, (label, color)) in zip(positions, INTERVENTIONS.items()):
        for alpha, offset, opacity in ((0.5, height / 1.7, 0.42), (0.0, -height / 1.7, 0.96)):
            values = subset.loc[(subset.intervention == code) & np.isclose(subset.alpha, alpha)]
            mean, low, high = date_clustered_mean_ci(values, rng, multiplier)
            ax.barh(position + offset, mean, height=height, color=color, alpha=opacity, zorder=2)
            ax.errorbar(mean, position + offset, xerr=[[mean - low], [high - mean]], fmt="none",
                        ecolor="#333333", elinewidth=1.15, capsize=2.5, zorder=3)
            pad = 0.025 * max(np.ptp(subset.actual_delta.to_numpy() * multiplier), 1e-9)
            long_bar = abs(mean) > 4.0 * pad
            if long_bar:
                text_x = mean - np.sign(mean) * pad
                text_ha = "right" if mean > 0 else "left"
                text_color = "white" if alpha == 0.0 else "#222222"
            else:
                text_x = mean + np.sign(mean if mean != 0 else 1) * pad
                text_ha = "left" if mean >= 0 else "right"
                text_color = "#222222"
            ax.text(text_x, position + offset, f"{mean:+.1f}", ha=text_ha, va="center",
                    color=text_color, fontsize=14.0, fontweight="bold")
    ax.axvline(0, color="#606060", lw=0.95, ls="--", zorder=1)
    ax.set_yticks(positions, [SHORT_LABELS[code] for code in INTERVENTIONS])
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold", pad=8)
    style_axis(ax)


def plot_aggregate_calibration(ax: plt.Axes, validation: pd.DataFrame, target: str, title: str, summary: pd.DataFrame) -> None:
    subset = validation.loc[validation.target == target]
    multiplier = 1000 if target == "co2" else 1
    rng = np.random.default_rng(61 if target == "cost" else 62)
    plotted = []
    for code, (label, color) in INTERVENTIONS.items():
        for alpha, marker in ((0.5, "o"), (0.0, "D")):
            block = subset.loc[(subset.intervention == code) & np.isclose(subset.alpha, alpha)]
            actual = block.actual_delta.to_numpy() * multiplier
            predicted = block.predicted_delta.to_numpy() * multiplier
            draws = rng.integers(0, len(block), size=(800, len(block)))
            actual_boot = actual[draws].mean(axis=1)
            predicted_boot = predicted[draws].mean(axis=1)
            x = float(predicted.mean()); y = float(actual.mean())
            xlow, xhigh = np.quantile(predicted_boot, [0.025, 0.975])
            ylow, yhigh = np.quantile(actual_boot, [0.025, 0.975])
            ax.errorbar(x, y, xerr=[[x - xlow], [xhigh - x]], yerr=[[y - ylow], [yhigh - y]],
                        fmt=marker, color=color, markeredgecolor="white", markeredgewidth=0.7,
                        ms=8.5, capsize=2.5, lw=1.2, zorder=3)
            ax.annotate(code, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=13.0, color="#333333")
            plotted.extend([x, y])
    lower, upper = min(plotted), max(plotted)
    margin = 0.12 * max(upper - lower, 1e-9)
    lower, upper = lower - margin, upper + margin
    ax.plot([lower, upper], [lower, upper], color="#333333", lw=1.1, ls="--", zorder=1)
    metric = summary.loc[summary.target == target].iloc[0]
    ax.text(0.04, 0.95, f"Daily $R^2$ = {metric.counterfactual_r2:.2f}\nDirection = {metric.direction_agreement_pct:.0f}%",
            transform=ax.transAxes, ha="left", va="top", fontsize=14.0)
    ax.set_xlim(lower, upper); ax.set_ylim(lower, upper)
    ax.set_xlabel("Surrogate mean $\Delta$")
    ax.set_ylabel("PyPSA mean $\Delta$")
    ax.set_title(title, fontweight="bold")
    style_axis(ax)


def plot_effect_heatmap(ax: plt.Axes, validation: pd.DataFrame, target: str, title: str) -> None:
    multiplier = 1000 if target == "co2" else 1
    zero = validation.loc[(validation.target == target) & np.isclose(validation.alpha, 0.0)].copy()
    table = zero.pivot_table(index="model", columns="intervention", values="actual_delta", aggfunc="mean")
    table = table.loc[list(MODELS), list(INTERVENTIONS)] * multiplier
    limit = float(np.abs(table.to_numpy()).max())
    image = ax.imshow(table.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(INTERVENTIONS)), list(INTERVENTIONS))
    ax.set_yticks(np.arange(len(MODELS)), MODELS)
    ax.set_title(title, fontweight="bold")
    for row in range(len(MODELS)):
        for column in range(len(INTERVENTIONS)):
            value = table.iloc[row, column]
            ax.text(column, row, f"{value:+.1f}", ha="center", va="center",
                    color="white" if abs(value) > 0.52 * limit else "#222222", fontsize=10.5, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return image


def plot_model_dots(ax: plt.Axes, validation: pd.DataFrame, target: str, title: str, xlabel: str, show_y: bool) -> None:
    subset = validation.loc[(validation.target == target) & np.isclose(validation.alpha, 0.0)]
    multiplier = 1000 if target == "co2" else 1
    positions = np.arange(len(INTERVENTIONS))[::-1]
    offsets = np.linspace(0.24, -0.24, len(MODELS))
    rng = np.random.default_rng(71 if target == "cost" else 72)
    for offset, model in zip(offsets, MODELS):
        for position, code in zip(positions, INTERVENTIONS):
            values = subset.loc[(subset.model == model) & (subset.intervention == code)]
            mean, low, high = date_clustered_mean_ci(values, rng, multiplier)
            ax.errorbar(mean, position + offset, xerr=[[mean - low], [high - mean]], fmt="o",
                        color=MODEL_COLORS[model], markeredgecolor="white", markeredgewidth=0.6,
                        ms=7.0, capsize=2.2, lw=1.25, zorder=3)
    ax.axvline(0, color="#606060", lw=0.95, ls="--", zorder=1)
    ax.set_yticks(positions, [SHORT_LABELS[code] for code in INTERVENTIONS] if show_y else [])
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold", pad=7)
    style_axis(ax)


def plot_absolute_reductions(
    ax_cost: plt.Axes,
    ax_co2: plt.Axes,
    uncertainty: pd.DataFrame,
) -> None:
    """Show annual absolute reductions with paired seven-day block-bootstrap CIs."""
    lookup = uncertainty.set_index(["model", "objective"])
    positions = np.arange(len(MODELS))
    for ax, objective, color, title, ylabel, decimals in (
        (ax_cost, "Total operating cost", OUTCOME_COLORS["total_cost"],
         "Annual total-cost reduction", "Reduction (bn CNY)", 2),
        (ax_co2, "Direct CO2", OUTCOME_COLORS["co2"],
         "Annual CO$_2$ reduction", "Reduction (Mt)", 1),
    ):
        block = lookup.loc[[(model, objective) for model in MODELS]]
        values = block.annual_reduction.to_numpy(dtype=float)
        lows = block.ci95_low.to_numpy(dtype=float)
        highs = block.ci95_high.to_numpy(dtype=float)
        errors = np.vstack([values - lows, highs - values])
        bars = ax.bar(positions, values, width=0.58, color=color, edgecolor="none", zorder=3)
        ax.errorbar(positions, values, yerr=errors, fmt="none", ecolor="#333333",
                    elinewidth=1.35, capsize=4.0, capthick=1.35, zorder=4)
        lower = min(0.0, float(lows.min()))
        span = max(float(highs.max() - lower), 1e-9)
        upper = float(highs.max() + 0.18 * span)
        ax.set_ylim(lower - 0.04 * span, upper)
        for bar, value, high in zip(bars, values, highs):
            ax.text(bar.get_x() + bar.get_width() / 2, high + span * 0.035,
                    f"{value:.{decimals}f}", ha="center", va="bottom",
                    fontsize=15.0, fontweight="bold", color="#222222")
        ax.set_xticks(positions, MODELS)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", pad=8)
        style_axis(ax, grid=False)
        ax.grid(axis="y", color="#E7EBEF", linewidth=0.7, zorder=0)


def make_figure(
    validation: pd.DataFrame,
    summary: pd.DataFrame,
    dynamic_uncertainty: pd.DataFrame,
    figure_dir: Path,
) -> tuple[Path, Path]:
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 17.5, "axes.titlesize": 20.0,
        "axes.labelsize": 18.0, "xtick.labelsize": 16.0, "ytick.labelsize": 16.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(18.2, 18.0))
    outer = GridSpec(4, 12, figure=fig, height_ratios=[0.92, 1.0, 0.92, 0.72],
                     hspace=0.64, wspace=0.78)

    # a | direct operational effects; this replaces the former schematic panel.
    spec_a = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0, :], wspace=0.28)
    axes_a = [fig.add_subplot(spec_a[0, index]) for index in range(3)]
    plot_effect_bars(axes_a[0], validation, "total_cost", "Total cost", "$\Delta$ total cost (M CNY d$^{-1}$)")
    plot_effect_bars(axes_a[1], validation, "balancing_cost", "Balancing cost", "$\Delta$ balancing cost (M CNY d$^{-1}$)")
    plot_effect_bars(axes_a[2], validation, "co2", "CO$_2$ emissions", "$\Delta$ CO$_2$ (kt d$^{-1}$)")
    for ax in axes_a[1:]:
        ax.set_yticklabels([])
    ax_a1 = axes_a[0]
    panel_label(ax_a1, "a", x=-0.16, y=1.15)
    strength_handles = [
        plt.Line2D([0], [0], marker="s", ls="none", markerfacecolor="#777777", alpha=0.42,
                   markeredgecolor="none", markersize=11, label="50% cut"),
        plt.Line2D([0], [0], marker="s", ls="none", markerfacecolor="#777777", alpha=0.96,
                   markeredgecolor="none", markersize=11, label="Removed"),
    ]
    ax_a1.legend(handles=strength_handles, loc="upper center", bbox_to_anchor=(1.70, -0.27),
                  ncol=2, frameon=False, fontsize=15.0)

    # b | eight scenario-level means instead of a dense cloud of daily points.
    spec_b = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1, :], wspace=0.32)
    axes_b = [fig.add_subplot(spec_b[0, index]) for index in range(3)]
    plot_aggregate_calibration(axes_b[0], validation, "total_cost", "Total-cost validation", summary)
    plot_aggregate_calibration(axes_b[1], validation, "balancing_cost", "Balancing-cost validation", summary)
    plot_aggregate_calibration(axes_b[2], validation, "co2", "CO$_2$ validation", summary)
    ax_b1 = axes_b[0]
    panel_label(ax_b1, "b", x=-0.24, y=1.14)
    level_handles = [
        plt.Line2D([0], [0], marker="o", ls="none", color="#666666", markersize=8, label="50% cut"),
        plt.Line2D([0], [0], marker="D", ls="none", color="#666666", markersize=8, label="Removed"),
    ]
    ax_b1.legend(handles=level_handles, loc="upper center", bbox_to_anchor=(1.70, -0.18),
                  ncol=2, frameon=False, fontsize=14.5)

    # c | model heterogeneity as grouped points and confidence intervals.
    spec_c = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2, :], wspace=0.30)
    axes_c = [fig.add_subplot(spec_c[0, index]) for index in range(3)]
    plot_model_dots(axes_c[0], validation, "total_cost", "Total cost", "$\Delta$ total cost\n(M CNY d$^{-1}$)", True)
    plot_model_dots(axes_c[1], validation, "balancing_cost", "Balancing cost", "$\Delta$ balancing cost\n(M CNY d$^{-1}$)", False)
    plot_model_dots(axes_c[2], validation, "co2", "CO$_2$ emissions", "$\Delta$ CO$_2$\n(kt d$^{-1}$)", False)
    ax_c1, ax_c2 = axes_c[0], axes_c[2]
    model_handles = [plt.Line2D([0], [0], marker="o", ls="none", color=MODEL_COLORS[model],
                                markersize=7.5, label=model) for model in MODELS]
    ax_c2.legend(handles=model_handles, loc="lower left", bbox_to_anchor=(0.01, 0.015),
                  ncol=1, frameon=False, fontsize=12.5, columnspacing=0.8,
                  handletextpad=0.30, borderaxespad=0.0)
    panel_label(ax_c1, "c", x=-0.34, y=1.15)

    # d | only the absolute annual reductions from dynamic adjustment.
    spec_d = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3, :], wspace=0.26)
    axes_d = [fig.add_subplot(spec_d[0, index]) for index in range(2)]
    plot_absolute_reductions(axes_d[0], axes_d[1], dynamic_uncertainty)
    panel_label(axes_d[0], "d", x=-0.15, y=1.13)

    fig.subplots_adjust(left=0.065, right=0.985, top=0.965, bottom=0.075)
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "05_val.png"
    pdf = figure_dir / "05_val.pdf"
    fig.savefig(png, dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def run(days_per_season: int, force: bool) -> None:
    logging.getLogger("pypsa").setLevel(logging.ERROR)
    logging.getLogger("linopy").setLevel(logging.ERROR)
    features = pd.read_csv(FEATURE_FILE, parse_dates=["date"])
    selected = select_representative_days(features, days_per_season)
    selected_days = selected.day_id.astype(int).tolist()
    selected.to_csv(RESULTS / "05_sampled_days.csv", index=False, float_format="%.8f")
    columns, surrogate_models = fit_surrogates(features)

    dispatch_module = load_dispatch_module()
    data = dispatch_module.inputs()
    factors = pd.read_csv(dispatch_module.FACTOR / "carrier_operational_co2_factors.csv").set_index("carrier").factor_t_co2_per_mwh
    provinces = list(data["provinces"])
    truth_full = data["wind"][provinces].to_numpy(dtype=float).reshape(365, 24, len(provinces))
    selected_times = pd.DatetimeIndex(np.concatenate([
        data["times"][day_id * 24:(day_id + 1) * 24].to_numpy() for day_id in selected_days
    ]))
    dispatch = pd.read_csv(DISPATCH_FILE, parse_dates=["timestamp_beijing"])
    raw = raw_daily_outcomes(dispatch, selected_days)

    cache_file = RESULTS / "05_counterfactual_daily_cache.csv"
    if cache_file.exists() and not force:
        cached = pd.read_csv(cache_file)
        if "total_operating_cost_million_cny" not in cached.columns:
            print("legacy cache lacks total operating cost; rebuilding counterfactual dispatch", flush=True)
            cached = pd.DataFrame()
    else:
        cached = pd.DataFrame()
    outcome_frames = [raw]
    feature_frames = []
    scenarios = [(code, alpha) for code in INTERVENTIONS for alpha in (0.5, 0.0)]

    for model in MODELS:
        prediction_full = data["scenarios"][model][provinces].to_numpy(dtype=float).reshape(365, 24, len(provinces))
        truth = truth_full[selected_days]
        prediction = prediction_full[selected_days]
        components = decompose_error(prediction, truth)
        capacity = data["caps"].loc[provinces, "wind"].to_numpy(dtype=float)
        for intervention, alpha in scenarios:
            scenario = f"{intervention}{int(alpha * 100):02d}"
            exists = (not cached.empty and ((cached.model == model) & (cached.scenario == scenario)).sum() == len(selected_days))
            cf_prediction, clipped_share = counterfactual_prediction(truth, components, intervention, alpha, capacity)
            cf_features = extract_features(model, selected_days, cf_prediction, truth, provinces)
            cf_features["scenario"] = scenario
            cf_features["intervention"] = intervention
            cf_features["alpha"] = alpha
            feature_frames.append(cf_features)
            if exists:
                print(f"cache {model:11s} {scenario}", flush=True)
                outcome_frames.append(cached.loc[(cached.model == model) & (cached.scenario == scenario)].copy())
                continue
            print(f"solve {model:11s} {scenario}: {len(selected_times)} h", flush=True)
            forecast = pd.DataFrame(cf_prediction.reshape(-1, len(provinces)), index=selected_times, columns=provinces)
            scenario_data = dict(data)
            scenario_data["scenarios"] = {model: forecast}
            hourly = dispatch_module.chunk(model, selected_times, scenario_data, factors)
            result = aggregate_counterfactual(hourly, model, selected_days, scenario, intervention, alpha, clipped_share)
            outcome_frames.append(result)
            cached = pd.concat([cached, result], ignore_index=True)
            cached.to_csv(cache_file, index=False, float_format="%.8f")

    outcomes = pd.concat(outcome_frames, ignore_index=True)
    cf_features = pd.concat(feature_frames, ignore_index=True)
    outcomes.to_csv(RESULTS / "05_counterfactual_daily.csv", index=False, float_format="%.8f")
    cf_features.to_csv(RESULTS / "05_counterfactual_features.csv", index=False, float_format="%.8f")
    validation = build_validation(outcomes, cf_features, features, columns, surrogate_models)
    validation.to_csv(RESULTS / "05_counterfactual_validation.csv", index=False, float_format="%.8f")
    summary = validation_summary(validation)
    summary.to_csv(RESULTS / "05_counterfactual_summary.csv", index=False, float_format="%.8f")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "design": "four-component error decomposition; 50% reduction and complete removal",
        "counterfactual_outcomes": {
            "total_cost": "total operating cost (realised fuel cost plus balancing cost)",
            "balancing_cost": "balancing cost",
            "co2": "direct operational CO2 emissions",
        },
        "sample": f"{days_per_season} severity-stratified operational days per season",
        "sampled_days": len(selected_days), "models": list(MODELS),
        "dispatch": "same two-stage PyPSA model as main_dispatch.py",
        "figure_uncertainty": "1000 date-clustered bootstrap replicates; dates resampled within season with all model and intervention rows retained",
        "interventions": {key: label for key, (label, _) in INTERVENTIONS.items()},
        "random_seed": 42,
    }
    (RESULTS / "05_counterfactual_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not DYNAMIC_2022_FILE.exists():
        raise FileNotFoundError(
            f"Missing 2022 dynamic-linear calibration: {DYNAMIC_2022_FILE}. "
            "Run 05_dynamic_linear_2022.py --stage summarize-dispatch first."
        )
    if not DYNAMIC_2022_DAILY_FILE.exists():
        raise FileNotFoundError(
            f"Missing paired daily dispatch results: {DYNAMIC_2022_DAILY_FILE}. "
            "Run 05_dynamic_linear_2022.py --stage summarize-dispatch first."
        )
    dynamic = pd.read_csv(DYNAMIC_2022_FILE)
    dynamic_absolute = dynamic.copy()
    dynamic_absolute["absolute_reduction"] = dynamic_absolute.raw_value - dynamic_absolute.dynamic_value
    dynamic_absolute.to_csv(
        RESULTS / "05_dynamic_absolute_reductions.csv", index=False, float_format="%.8f"
    )
    paired_daily, dynamic_uncertainty = dynamic_reduction_uncertainty(
        pd.read_csv(DYNAMIC_2022_DAILY_FILE)
    )
    paired_daily.to_csv(
        RESULTS / "05_dynamic_daily_reductions.csv", index=False, float_format="%.8f"
    )
    dynamic_uncertainty.to_csv(
        RESULTS / "05_dynamic_reduction_uncertainty.csv", index=False, float_format="%.8f"
    )
    dynamic_manifest_path = DYNAMIC_2022_FILE.parent / "manifest.json"
    if dynamic_manifest_path.exists():
        dynamic_manifest = json.loads(dynamic_manifest_path.read_text(encoding="utf-8"))
        dynamic_manifest["annual_reduction_uncertainty"] = {
            "method": "paired circular moving-block bootstrap of daily Raw-minus-dynamic effects",
            "block_days": 7,
            "bootstrap_repeats": 5000,
            "confidence_interval": "95% percentile interval",
            "boundary_handling": "two partial calendar-boundary days combined into one operational day",
            "daily_output": "../05_dynamic_daily_reductions.csv",
            "summary_output": "../05_dynamic_reduction_uncertainty.csv",
        }
        dynamic_manifest_path.write_text(json.dumps(dynamic_manifest, indent=2), encoding="utf-8")
    png, pdf = make_figure(validation, summary, dynamic_uncertainty, FIGURES)
    print(summary.to_string(index=False), flush=True)
    print(f"saved figure: {png}\nsaved figure: {pdf}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days-per-season", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="ignore cached dispatch results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.days_per_season, args.force)

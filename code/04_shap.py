"""Spatiotemporal SHAP analysis of forecast-driven operating impacts.

The analysis unit is one model-operation-day (24 consecutive hours).  Four
wind forecasting models over the 8,760-hour operational year yield 1,460
samples. TreeSHAP separately explains daily total-operating-cost increment,
balancing cost, and direct-CO2 change versus perfect foresight using national
temporal and province/region spatial error features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import explained_variance_score, r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
FORECAST = ROOT / "results" / "wind_forecast_2020_all_models.csv"
ACTUAL = ROOT / "dataset" / "provincial_hourly_2018_2023" / "provincial_hourly_wind_2018_2023.csv"
DISPATCH = ROOT / "results" / "pypsa_dispatch_2020_all_models.csv"
PERFECT = ROOT / "results" / "pypsa_two_stage_2020" / "perfect_forecast_hourly.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figure"

MODELS = ("CNN", "LSTM", "Transformer", "DNN")
MODEL_KEYS = {"cnn": "CNN", "lstm": "LSTM", "transformer": "Transformer", "dnn": "DNN"}
SEASONS = ("Winter", "Spring", "Summer", "Autumn")
SEASON_COLORS = {"Winter": "#55A6D9", "Spring": "#59B85D", "Summer": "#F4C34E", "Autumn": "#C77DB5"}
REFERENCE_CMAP = LinearSegmentedColormap.from_list(
    "classic_red_blue", ["#2166AC", "#F7F7F7", "#B2182B"]
)
REFERENCE_BLUE = "#2166AC"
REFERENCE_RED = "#B2182B"
REFERENCE_LINE = "#5A5A5A"
GROUP_COLORS = {
    "Temporal bias": "#ED5675",
    "Temporal variability": "#F2A34A",
    "Spatial bias": "#55B7C9",
    "Spatial variability": "#67B96B",
}
REGIONS = ("N", "E", "NW", "NE", "SW", "C", "S")
PROVINCE_REGION = {
    "Beijing": "N", "Tianjin": "N", "Hebei": "N", "Shanxi": "N", "InnerMongolia": "N",
    "Shanghai": "E", "Jiangsu": "E", "Zhejiang": "E", "Anhui": "E", "Fujian": "E",
    "Jiangxi": "E", "Shandong": "E", "Shaanxi": "NW", "Gansu": "NW", "Qinghai": "NW",
    "Ningxia": "NW", "Xinjiang": "NW", "Liaoning": "NE", "Jilin": "NE", "Heilongjiang": "NE",
    "Chongqing": "SW", "Sichuan": "SW", "Guizhou": "SW", "Yunnan": "SW", "Tibet": "SW",
    "Henan": "C", "Hubei": "C", "Hunan": "C", "Guangdong": "S", "Guangxi": "S", "Hainan": "S",
}

DISPLAY_NAMES = {
    "underforecast_bias_gw": "Underforecast bias",
    "overforecast_bias_gw": "Overforecast bias",
    "maximum_hourly_error_gw": "Maximum hourly error",
    "temporal_error_sd_gw": "Temporal error SD",
    "ramp_error_gw": "Ramp error",
    "error_persistence_h": "Error persistence",
    "provincial_abs_bias_gw": "Provincial |bias|",
    "spatial_error_sd_gw": "Spatial error SD",
    "spatial_p90_gw": "Spatial P90",
    "spatial_concentration": "Spatial concentration",
}


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def longest_sign_run(values: np.ndarray) -> int:
    signs = np.sign(values)
    signs[signs == 0] = 1
    changes = np.flatnonzero(np.r_[True, signs[1:] != signs[:-1], True])
    return int(np.diff(changes).max())


def load_aligned_hourly() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    forecast = pd.read_csv(FORECAST, parse_dates=["valid_time_beijing"])
    metadata = {"model", "forecast_origin_beijing", "lead_hour", "valid_time_beijing"}
    provinces = [column for column in forecast.columns if column not in metadata]
    if set(provinces) != set(PROVINCE_REGION):
        raise ValueError("Province columns do not match the regional mapping")

    valid_times = pd.DatetimeIndex(sorted(forecast.valid_time_beijing.unique()))
    valid_times = valid_times[~((valid_times.month == 2) & (valid_times.day == 29))]
    if len(valid_times) != 8_760:
        raise ValueError(f"Expected 8,760 valid hours, found {len(valid_times)}")

    actual = pd.read_csv(ACTUAL, parse_dates=["timestamp_beijing"]).set_index("timestamp_beijing")
    actual = actual.reindex(valid_times)[provinces]
    if actual.isna().any().any():
        raise ValueError("Observed provincial wind is incomplete")

    blocks = []
    for key, display in MODEL_KEYS.items():
        model = forecast.loc[forecast.model == key].set_index("valid_time_beijing").reindex(valid_times)
        if model[provinces].isna().any().any():
            raise ValueError(f"Forecast is incomplete for {display}")
        block = model[provinces].copy()
        block.insert(0, "hour_in_day", np.arange(len(block)) % 24)
        block.insert(0, "day_id", np.arange(len(block)) // 24)
        block.insert(0, "model", display)
        block.insert(0, "valid_time", valid_times)
        blocks.append(block.reset_index(drop=True))
    return pd.concat(blocks, ignore_index=True), actual, provinces


def build_features() -> pd.DataFrame:
    forecast, actual, provinces = load_aligned_hourly()
    operation_dates = pd.date_range("2020-01-01", periods=365, freq="D")
    rows: list[dict[str, float | int | str]] = []

    for model in MODELS:
        model_frame = forecast.loc[forecast.model == model].sort_values("valid_time")
        prediction = model_frame[provinces].to_numpy(dtype=float).reshape(365, 24, len(provinces))
        truth = actual.to_numpy(dtype=float).reshape(365, 24, len(provinces))
        error = prediction - truth

        for day_id, date in enumerate(operation_dates):
            e = error[day_id]
            national = e.sum(axis=1) / 1_000.0
            predicted_national = prediction[day_id].sum(axis=1) / 1_000.0
            actual_national = truth[day_id].sum(axis=1) / 1_000.0
            province_bias = e.mean(axis=0) / 1_000.0
            province_abs_energy = np.abs(e).sum(axis=0)
            shares = province_abs_energy / max(province_abs_energy.sum(), 1e-12)

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
                indices = [index for index, province in enumerate(provinces) if PROVINCE_REGION[province] == region]
                regional_hourly = e[:, indices].sum(axis=1) / 1_000.0
                regional_bias = e[:, indices].mean(axis=0) / 1_000.0
                row[f"region_{region}_under_gw"] = float(np.clip(-regional_hourly, 0, None).mean())
                row[f"region_{region}_over_gw"] = float(np.clip(regional_hourly, 0, None).mean())
                row[f"region_{region}_dispersion_gw"] = float(regional_bias.std(ddof=0))
            rows.append(row)

    features = pd.DataFrame(rows)
    targets = build_targets()
    merged = features.merge(targets, on=["model", "day_id"], how="left", validate="one_to_one")
    required_targets = ["total_cost_change_million_cny", "balancing_cost_million_cny", "co2_change_mt"]
    if merged[required_targets].isna().any().any():
        raise ValueError("Daily targets do not align with the feature table")
    return merged


def build_targets() -> pd.DataFrame:
    dispatch = pd.read_csv(DISPATCH, parse_dates=["timestamp_beijing"])
    perfect = pd.read_csv(PERFECT, parse_dates=["timestamp_beijing"]).set_index("timestamp_beijing")
    rows = []
    for model in MODELS:
        block = dispatch.loc[dispatch.model == model].sort_values("timestamp_beijing").reset_index(drop=True)
        if len(block) != 8_760:
            raise ValueError(f"Dispatch for {model} contains {len(block)} hours")
        baseline_co2 = block.timestamp_beijing.map(perfect.direct_co2_t)
        baseline_total_cost = block.timestamp_beijing.map(perfect.total_operating_cost_cny)
        if baseline_co2.isna().any() or baseline_total_cost.isna().any():
            raise ValueError("Perfect-forecast outcomes do not cover all dispatch hours")
        for day_id in range(365):
            selection = slice(day_id * 24, (day_id + 1) * 24)
            day = block.iloc[selection]
            rows.append({
                "model": model,
                "day_id": day_id,
                "total_cost_change_million_cny": float(
                    (day.total_operating_cost_cny.to_numpy() - baseline_total_cost.iloc[selection].to_numpy()).sum() / 1e6
                ),
                "balancing_cost_million_cny": float(day.balancing_cost_cny.sum() / 1e6),
                "co2_change_mt": float((day.direct_co2_t.to_numpy() - baseline_co2.iloc[selection].to_numpy()).sum() / 1e6),
            })
    return pd.DataFrame(rows)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "model", "day_id", "date", "season", "total_cost_change_million_cny",
        "balancing_cost_million_cny", "co2_change_mt",
    }
    return [column for column in frame.columns if column not in excluded]


def feature_group(name: str) -> str:
    if name in {"underforecast_bias_gw", "overforecast_bias_gw"}:
        return "Temporal bias"
    if name in {"maximum_hourly_error_gw", "temporal_error_sd_gw", "ramp_error_gw", "error_persistence_h"}:
        return "Temporal variability"
    if name == "provincial_abs_bias_gw" or name.endswith("_under_gw") or name.endswith("_over_gw"):
        return "Spatial bias"
    return "Spatial variability"


def new_model(seed: int = 42) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=450,
        max_depth=3,
        learning_rate=0.025,
        min_child_weight=5,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_alpha=0.04,
        reg_lambda=1.2,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=8,
    )


def fit_and_explain(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    columns = feature_columns(frame)
    X = frame[columns]
    validation_rows = []
    outputs: dict[str, dict[str, object]] = {}
    targets = {
        "total_cost": "total_cost_change_million_cny",
        "balancing_cost": "balancing_cost_million_cny",
        "co2": "co2_change_mt",
    }

    for target_key, target_column in targets.items():
        y = frame[target_column]
        for held_season in SEASONS:
            test = frame.season == held_season
            train = ~test
            model = new_model()
            model.fit(X.loc[train], y.loc[train])
            train_prediction = model.predict(X.loc[train])
            test_prediction = model.predict(X.loc[test])
            validation_rows.append({
                "target": target_key,
                "held_out_season": held_season,
                "train_samples": int(train.sum()),
                "test_samples": int(test.sum()),
                "train_r2": float(r2_score(y.loc[train], train_prediction)),
                "held_out_r2": float(r2_score(y.loc[test], test_prediction)),
                "held_out_explained_variance": float(explained_variance_score(y.loc[test], test_prediction)),
            })

        final_model = new_model()
        final_model.fit(X, y)
        explanation = shap.TreeExplainer(final_model)(X)
        outputs[target_key] = {
            "model": final_model,
            "values": np.asarray(explanation.values, dtype=float),
            "base_values": np.asarray(explanation.base_values, dtype=float),
            "features": X,
            "target": y,
            "columns": columns,
        }
    return pd.DataFrame(validation_rows), outputs


def save_outputs(frame: pd.DataFrame, validation: pd.DataFrame, explained: dict[str, dict[str, object]], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "04_shap_features.csv", index=False, float_format="%.8f")
    validation.to_csv(output_dir / "04_shap_validation.csv", index=False, float_format="%.8f")

    grouped_rows = []
    region_rows = []
    for target, result in explained.items():
        values = result["values"]
        columns = result["columns"]
        shap_frame = pd.DataFrame(values, columns=columns)
        shap_frame.insert(0, "date", frame.date)
        shap_frame.insert(0, "season", frame.season)
        shap_frame.insert(0, "day_id", frame.day_id)
        shap_frame.insert(0, "model", frame.model)
        shap_frame.to_csv(output_dir / f"04_shap_values_{target}.csv", index=False, float_format="%.8f")

        total = np.abs(values).sum()
        for group in GROUP_COLORS:
            indices = [index for index, name in enumerate(columns) if feature_group(name) == group]
            contribution = float(np.abs(values[:, indices]).sum())
            grouped_rows.append({"target": target, "group": group, "mean_abs_shap": contribution / len(frame), "share_pct": 100 * contribution / total})

        for season in SEASONS:
            mask = frame.season.eq(season).to_numpy()
            for region in REGIONS:
                indices = [index for index, name in enumerate(columns) if name.startswith(f"region_{region}_")]
                region_rows.append({
                    "target": target,
                    "season": season,
                    "region": region,
                    "mean_abs_shap": float(np.abs(values[mask][:, indices]).sum(axis=1).mean()),
                })

    grouped = pd.DataFrame(grouped_rows)
    regional = pd.DataFrame(region_rows)
    grouped.to_csv(output_dir / "04_shap_grouped_contributions.csv", index=False, float_format="%.8f")
    regional.to_csv(output_dir / "04_shap_region_season.csv", index=False, float_format="%.8f")
    manifest = {
        "samples": len(frame),
        "operation_days": 365,
        "models": list(MODELS),
        "features": feature_columns(frame),
        "targets": {
            "total_cost": "daily total operating cost change versus perfect forecast (million CNY)",
            "balancing_cost": "daily balancing cost (million CNY)",
            "co2": "daily direct CO2 change versus perfect forecast (Mt)",
        },
        "validation": "leave-one-season-out; all four models for a date remain in the same fold",
        "interpretation": "SHAP associations are predictive explanations, not causal effects",
        "random_seed": 42,
    }
    (output_dir / "04_shap_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return grouped, regional


def style_axis(ax: plt.Axes, grid: bool = True) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#555555")
        spine.set_linewidth(0.72)
    ax.tick_params(width=0.72, length=3.2, color="#555555")
    if grid:
        ax.grid(True, color="#E4E4E4", linewidth=0.58, alpha=0.82, zorder=0)


def panel_label(ax: plt.Axes, label: str, x: float = -0.18, y: float = 1.12) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=20, fontweight="bold", ha="left", va="top")


TARGET_SPECS = (
    ("total_cost", "Total cost", "M CNY day$^{-1}$"),
    ("balancing_cost", "Balancing cost", "M CNY day$^{-1}$"),
    ("co2", "CO$_2$ emissions", "Mt day$^{-1}$"),
)


def plot_validation(axes: list[plt.Axes], validation: pd.DataFrame) -> None:
    metric_colors = {
        "Held-out $R^2$": REFERENCE_BLUE,
        "Explained variance": REFERENCE_RED,
    }
    for axis_index, (ax, (target, title, _)) in enumerate(zip(axes, TARGET_SPECS)):
        subset = validation.loc[validation.target == target].set_index("held_out_season").loc[list(SEASONS)]
        positions = np.arange(len(SEASONS))[::-1]
        r2_values = subset.held_out_r2.to_numpy()
        variance_values = subset.held_out_explained_variance.to_numpy()
        left = 0.70
        height = 0.28
        ax.barh(positions + height / 2, r2_values - left, left=left, height=height,
                color=metric_colors["Held-out $R^2$"], label="Held-out $R^2$", zorder=2)
        ax.barh(positions - height / 2, variance_values - left, left=left, height=height,
                color=metric_colors["Explained variance"], label="Explained variance", zorder=2)
        for position, r2_value, variance_value in zip(positions, r2_values, variance_values):
            ax.text(r2_value - 0.006, position + height / 2, f"{r2_value:.2f}",
                    ha="right", va="center", color="white", fontsize=11.5, fontweight="bold")
            ax.text(variance_value - 0.006, position - height / 2, f"{variance_value:.2f}",
                    ha="right", va="center", color="white", fontsize=11.5, fontweight="bold")
        ax.set_yticks(positions, SEASONS)
        ax.set_xlabel("Held-out validation score")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=7)
        ax.set_xlim(left, 1.005)
        ax.set_xticks([0.70, 0.80, 0.90, 1.00])
        if axis_index:
            ax.set_yticklabels([])
        style_axis(ax)


def display_name(name: str) -> str:
    if name in DISPLAY_NAMES:
        return DISPLAY_NAMES[name]
    if name.startswith("region_"):
        _, region, metric = name.split("_", 2)
        suffix = {"under_gw": "underforecast", "over_gw": "overforecast", "dispersion_gw": "dispersion"}[metric]
        return f"{region} regional {suffix}"
    return name


def plot_beeswarm(
    ax: plt.Axes,
    result: dict[str, object],
    xlabel: str,
    show_feature_legend: bool,
    seed: int,
    selected_names: list[str] | None = None,
) -> None:
    values = result["values"]
    features = result["features"]
    columns = result["columns"]
    if selected_names is None:
        importance = np.abs(values).mean(axis=0)
        selected = np.argsort(importance)[-6:][::-1]
    else:
        selected = np.asarray([columns.index(name) for name in selected_names], dtype=int)
    cmap = REFERENCE_CMAP
    rng = np.random.default_rng(seed)
    for position, index in enumerate(selected[::-1]):
        feature_values = features.iloc[:, index].to_numpy()
        lower, upper = np.quantile(feature_values, [0.05, 0.95])
        normalized = np.clip((feature_values - lower) / max(upper - lower, 1e-12), 0, 1)
        jitter = rng.normal(0, 0.105, len(values))
        ax.scatter(values[:, index], position + jitter, c=normalized, cmap=cmap,
                   s=9, alpha=0.52, edgecolors="none", rasterized=True, zorder=2)
    ax.axvline(0, color=REFERENCE_LINE, lw=0.75)
    ax.set_yticks(np.arange(len(selected)), [display_name(columns[index]) for index in selected[::-1]])
    ax.set_xlabel(xlabel)
    style_axis(ax)
    if show_feature_legend:
        ax.text(0.64, 0.13, "Feature value", transform=ax.transAxes, fontsize=11.5)
        ax.scatter([0.72, 0.91], [0.055, 0.055], transform=ax.transAxes, c=[0, 1], cmap=cmap, s=24, clip_on=False)
        ax.text(0.75, 0.055, "Low", transform=ax.transAxes, va="center", fontsize=11.0)
        ax.text(0.94, 0.055, "High", transform=ax.transAxes, va="center", fontsize=11.0)


def plot_grouped(axes: list[plt.Axes], grouped: pd.DataFrame) -> None:
    centers = {"total_cost": "Total\ncost", "balancing_cost": "Balancing\ncost", "co2": "CO$_2$"}
    for ax, (target, _, _) in zip(axes, TARGET_SPECS):
        subset = grouped.loc[grouped.target == target].set_index("group").loc[list(GROUP_COLORS)]
        wedges, _ = ax.pie(
            subset.share_pct,
            colors=[GROUP_COLORS[group] for group in subset.index],
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.31, "edgecolor": "white", "linewidth": 1.0},
        )
        for wedge, value in zip(wedges, subset.share_pct):
            if value >= 5:
                angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
                ax.text(0.82 * np.cos(angle), 0.82 * np.sin(angle), f"{value:.0f}%", ha="center", va="center", color="white", fontsize=11.5, fontweight="bold")
        ax.text(0, 0, centers[target], ha="center", va="center", fontsize=13.5, fontweight="bold")
        ax.set_aspect("equal")


def plot_dependence(ax: plt.Axes, result: dict[str, object], feature: str, color_feature: str, ylabel: str, color_label: str, title: str) -> None:
    columns = result["columns"]
    feature_index = columns.index(feature)
    x = result["features"][feature].to_numpy()
    y = result["values"][:, feature_index]
    color = result["features"][color_feature].to_numpy()
    lower, upper = np.quantile(color, [0.03, 0.97])
    norm = Normalize(lower, upper)
    scatter = ax.scatter(x, y, c=color, cmap=REFERENCE_CMAP, norm=norm, s=11, alpha=0.60, edgecolors="none", rasterized=True, zorder=2)
    smooth = lowess(y, x, frac=0.24, it=1, return_sorted=True)
    ax.plot(smooth[:, 0], smooth[:, 1], color=REFERENCE_LINE, lw=2.2, zorder=3)
    ax.axhline(0, color="#666666", ls="--", lw=0.75)
    ax.set_xlabel(display_name(feature))
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=5)
    style_axis(ax)
    colorbar = plt.colorbar(scatter, ax=ax, orientation="horizontal", fraction=0.055, pad=0.20, aspect=22)
    colorbar.set_label(color_label, fontsize=11.0)
    colorbar.ax.tick_params(labelsize=10.0, length=2)


def plot_region_heatmap(ax: plt.Axes, regional: pd.DataFrame, target: str, title: str, cmap: str, show_y: bool) -> None:
    table = regional.loc[regional.target == target].pivot(index="region", columns="season", values="mean_abs_shap").loc[list(REGIONS), list(SEASONS)]
    image = ax.imshow(table.to_numpy(), cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(4), ["W", "Sp", "Su", "A"])
    ax.set_yticks(np.arange(len(REGIONS)), REGIONS if show_y else [])
    ax.set_xlabel("Season")
    if show_y:
        ax.set_ylabel("Region")
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=5)
    value_format = ".3f" if target == "co2" else ".2f"
    for row in range(len(REGIONS)):
        for column in range(4):
            ax.text(column, row, format(table.iloc[row, column], value_format), ha="center", va="center", fontsize=7.0, color="white" if table.iloc[row, column] > table.to_numpy().max() * 0.55 else "#222222")
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = plt.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.ax.tick_params(labelsize=7, length=2)
    colorbar.set_label("Mean |SHAP|", fontsize=8)


def plot_response(
    ax: plt.Axes,
    result: dict[str, object],
    feature: str,
    ylabel: str,
    title: str | None,
    show_xlabel: bool,
) -> None:
    """Compact dependence response for the complete target-feature matrix."""
    feature_index = result["columns"].index(feature)
    x = result["features"][feature].to_numpy()
    y = result["values"][:, feature_index]
    lower, upper = np.quantile(x, [0.03, 0.97])
    norm = Normalize(lower, upper)
    ax.scatter(x, y, s=8, c=x, cmap=REFERENCE_CMAP, norm=norm,
               alpha=0.34, edgecolors="none", rasterized=True, zorder=2)
    smooth = lowess(y, x, frac=0.24, it=1, return_sorted=True)
    ax.plot(smooth[:, 0], smooth[:, 1], color=REFERENCE_LINE, lw=2.0, zorder=3)
    ax.axhline(0, color="#777777", ls="--", lw=0.72)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(display_name(feature) if show_xlabel else "")
    if title:
        ax.set_title(title, fontsize=13.0, fontweight="bold", pad=5)
    style_axis(ax)


def make_figure(validation: pd.DataFrame, explained: dict[str, dict[str, object]], grouped: pd.DataFrame, regional: pd.DataFrame, figure_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 13.5, "axes.titlesize": 16.0,
        "axes.labelsize": 14.0, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(18.8, 14.2))
    outer = GridSpec(3, 1, figure=fig, height_ratios=[0.92, 1.15, 2.75], hspace=0.58)

    # a | seasonal held-out skill for all three operational outcomes.
    spec_a = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.30)
    axes_a = [fig.add_subplot(spec_a[0, index]) for index in range(3)]
    plot_validation(axes_a, validation)
    panel_label(axes_a[0], "a", x=-0.14, y=1.15)
    validation_handles, validation_labels = axes_a[0].get_legend_handles_labels()
    axes_a[0].legend(
        validation_handles, validation_labels, loc="upper left", bbox_to_anchor=(0.00, -0.20),
        ncol=2, frameon=False, fontsize=12.5, handlelength=1.25,
        handletextpad=0.45, columnspacing=1.05,
    )

    # b | comparable SHAP distributions for total cost, balancing cost and CO2.
    spec_b = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.34)
    axes_b = [fig.add_subplot(spec_b[0, index]) for index in range(3)]
    shared_features = [
        "underforecast_bias_gw", "maximum_hourly_error_gw", "overforecast_bias_gw",
        "provincial_abs_bias_gw", "temporal_error_sd_gw", "ramp_error_gw",
    ]
    for index, (ax, (target, title, unit)) in enumerate(zip(axes_b, TARGET_SPECS)):
        plot_beeswarm(ax, explained[target], f"SHAP ({unit})", index == 2, 42 + index, shared_features)
        ax.set_title(title, fontsize=16.0, fontweight="bold", pad=5)
        if index:
            ax.set_yticklabels([])
    panel_label(axes_b[0], "b", x=-0.14, y=1.14)

    # c | complete 3-outcome by 5-error-response matrix.
    response_features = [
        "underforecast_bias_gw", "overforecast_bias_gw", "maximum_hourly_error_gw",
        "provincial_abs_bias_gw", "temporal_error_sd_gw",
    ]
    response_titles = [
        "Underforecast bias", "Overforecast bias", "Maximum hourly error",
        "Provincial |bias|", "Temporal error SD",
    ]
    row_labels = {
        "total_cost": "Total-cost SHAP\n(M CNY day$^{-1}$)",
        "balancing_cost": "Balancing-cost SHAP\n(M CNY day$^{-1}$)",
        "co2": "CO$_2$ SHAP\n(Mt day$^{-1}$)",
    }
    spec_d = GridSpecFromSubplotSpec(3, 5, subplot_spec=outer[2], hspace=0.48, wspace=0.36)
    axes_d: list[list[plt.Axes]] = []
    for row, (target, _, _) in enumerate(TARGET_SPECS):
        row_axes = []
        for column, feature in enumerate(response_features):
            ax = fig.add_subplot(spec_d[row, column])
            plot_response(
                ax, explained[target], feature,
                row_labels[target] if column == 0 else "",
                response_titles[column] if row == 0 else None,
                show_xlabel=row == 2,
            )
            row_axes.append(ax)
        axes_d.append(row_axes)
    panel_label(axes_d[0][0], "c", x=-0.17, y=1.20)

    fig.subplots_adjust(left=0.065, right=0.988, top=0.972, bottom=0.075)
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "04_shap.png"
    pdf = figure_dir / "04_shap.pdf"
    fig.savefig(png, dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=RESULTS)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("building 365-day x 4-model spatiotemporal feature table", flush=True)
    frame = build_features()
    print(f"samples={len(frame)} features={len(feature_columns(frame))}", flush=True)
    validation, explained = fit_and_explain(frame)
    grouped, regional = save_outputs(frame, validation, explained, args.result_dir)
    png, pdf = make_figure(validation, explained, grouped, regional, args.figure_dir)
    print(validation.to_string(index=False), flush=True)
    print(f"saved figure: {png}")
    print(f"saved figure: {pdf}")


if __name__ == "__main__":
    main()

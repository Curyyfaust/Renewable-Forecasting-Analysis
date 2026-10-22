"""Leakage-free dynamic linear forecast calibration for the Figure 5 experiment.

Four forecasting models are trained using 2021 data only.  Their final-quarter
2021 validation residuals initialise two objective-specific weighted ridge
calibrators.  During 2022, coefficients are re-estimated once per forecast
origin using observations that were available at that origin.  No 2022
outcome is used to select a transformation or a hyperparameter.

This script intentionally creates numerical results only; it does not plot.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataloader import WindForecastDataset
from exp import NormalizedWindDataset, set_seed, train_model
from forecasting_2020 import forecast_frame, predict_model
from model import build_model


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "05_dynamic_linear_2022"
WIND_PATH = ROOT / "dataset" / "provincial_hourly_2018_2023" / "provincial_hourly_wind_2018_2023.csv"
WEATHER_PATH = ROOT / "dataset" / "weather_forecast_24h" / "weatherbench2_hres_china_2018_2022.npz"
MODELS = ("cnn", "lstm", "transformer", "dnn")
OBJECTIVES = ("cost", "co2")
DISPLAY = {"cnn": "CNN", "lstm": "LSTM", "transformer": "Transformer", "dnn": "DNN"}
FEATURES = (
    "intercept", "forecast_level", "forecast_ramp", "hour_sin", "hour_cos",
    "doy_sin", "doy_cos", "lagged_bias",
)
DISPATCH_MODULE = ROOT / "PyPSA-china" / "main_dispatch.py"


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def make_2021_loaders(
    batch_size: int, validation_fraction: float, seed: int
) -> tuple[DataLoader, DataLoader, dict[str, np.ndarray], pd.Timestamp]:
    """Create a chronological 2021 split with training-only normalisation."""
    dataset = WindForecastDataset(year=2021)
    validation_size = max(1, round(len(dataset) * validation_fraction))
    split = len(dataset) - validation_size
    last_training_hour = int(dataset.origin_hour_offsets[split - 1]) + 24
    training_wind = dataset.wind[: last_training_hour + 1]
    stats = {
        "wind_mean": training_wind.mean(axis=0),
        "wind_std": np.maximum(training_wind.std(axis=0), 1e-6),
        "weather_mean": dataset.forecast[:split].mean(axis=(0, 1, 2)),
        "weather_std": np.maximum(dataset.forecast[:split].std(axis=(0, 1, 2)), 1e-6),
    }
    train_set = NormalizedWindDataset(dataset, range(split), **stats)
    validation_set = NormalizedWindDataset(dataset, range(split, len(dataset)), **stats)
    common = dict(batch_size=batch_size, num_workers=0, pin_memory=torch.cuda.is_available())
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common)
    validation_loader = DataLoader(validation_set, shuffle=False, **common)
    validation_start = pd.Timestamp(dataset.origins[split])
    return train_loader, validation_loader, stats, validation_start


def load_daily_inputs(year: int, issue_hour: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load every daily origin, retaining cross-year history and target context."""
    wind = pd.read_csv(WIND_PATH, parse_dates=["timestamp_beijing"]).set_index("timestamp_beijing").sort_index()
    archive = np.load(WEATHER_PATH, allow_pickle=False)
    origins_all = pd.DatetimeIndex(archive["origin"])
    provinces = [str(value) for value in archive["provinces"].tolist()]
    selected = np.flatnonzero((origins_all.year == year) & (origins_all.hour == issue_hour))
    origins = origins_all[selected]
    expected = pd.date_range(f"{year}-01-01 {issue_hour:02d}:00", f"{year}-12-31 {issue_hour:02d}:00", freq="D")
    if not origins.equals(expected):
        raise ValueError(f"Weather archive does not contain all {year} daily origins at {issue_hour:02d}:00")
    histories, targets = [], []
    for origin in origins:
        history_index = pd.date_range(origin - pd.Timedelta(hours=71), origin, freq="h")
        target_index = pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="h")
        history = wind.reindex(history_index)[provinces]
        target = wind.reindex(target_index)[provinces]
        if history.isna().any().any() or target.isna().any().any():
            raise ValueError(f"Wind observations are incomplete around {origin}")
        histories.append(history.to_numpy(np.float32))
        targets.append(target.to_numpy(np.float32))
    weather = np.asarray(archive["forecast"][selected], dtype=np.float32)
    return origins.to_numpy("datetime64[ns]"), np.stack(histories), weather, np.stack(targets), provinces


def train_and_predict(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = OUT / "checkpoints"
    forecast_dir = OUT / "forecasts"
    checkpoint_dir.mkdir(exist_ok=True)
    forecast_dir.mkdir(exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_loader, stats, validation_start = make_2021_loaders(
        args.batch_size, args.validation_fraction, args.seed
    )
    np.savez(OUT / "2021_training_normalisation.npz", **stats)
    inputs = {year: load_daily_inputs(year, args.issue_hour) for year in (2021, 2022)}
    summary = []
    for model_name in MODELS:
        checkpoint = checkpoint_dir / f"{model_name}_2021_best.pt"
        history_path = checkpoint_dir / f"{model_name}_2021_history.csv"
        if not checkpoint.exists() or args.force_train:
            set_seed(args.seed)
            model = build_model(model_name)
            history = train_model(
                model, train_loader, val_loader, epochs=args.epochs,
                learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                device=device, checkpoint_path=checkpoint,
            )
            write_history(history_path, history)
        history = pd.read_csv(history_path)
        best = history.loc[history.val_mse.idxmin()]
        summary.append({
            "model": DISPLAY[model_name], "best_epoch": int(best.epoch),
            "validation_mse_mw2": best.val_mse, "validation_mae_mw": best.val_mae,
        })
        for year in (2021, 2022):
            origins, history_values, weather, actual, provinces = inputs[year]
            prediction = predict_model(
                model_name, checkpoint, history_values, weather,
                stats["wind_mean"], stats["wind_std"], stats["weather_mean"], stats["weather_std"],
                device, args.batch_size,
            )
            frame = forecast_frame(origins, prediction, provinces)
            actual_frame = forecast_frame(origins, actual, provinces)
            if year == 2021:
                keep = frame.forecast_origin_beijing >= validation_start
                frame = frame.loc[keep].reset_index(drop=True)
                actual_frame = actual_frame.loc[keep].reset_index(drop=True)
            frame.to_csv(forecast_dir / f"{model_name}_{year}_raw.csv", index=False, float_format="%.4f")
            actual_frame.to_csv(forecast_dir / f"observed_{year}.csv", index=False, float_format="%.4f")
            print(f"saved {model_name} {year}: {len(frame)} hourly forecasts", flush=True)
    pd.DataFrame(summary).to_csv(OUT / "2021_training_summary.csv", index=False, float_format="%.6f")


def design_matrix(
    forecast: np.ndarray,
    times: pd.DatetimeIndex,
    scale: float,
    lagged_bias: float,
) -> np.ndarray:
    level = forecast / scale
    ramp = np.r_[0.0, np.diff(forecast)] / scale
    hour_angle = 2 * np.pi * times.hour.to_numpy() / 24.0
    day_angle = 2 * np.pi * (times.dayofyear.to_numpy() - 1) / 365.25
    return np.column_stack([
        np.ones(len(times)), level, ramp, np.sin(hour_angle), np.cos(hour_angle),
        np.sin(day_angle), np.cos(day_angle), np.full(len(times), lagged_bias / scale),
    ])


def objective_weights(residual: np.ndarray, objective: str) -> np.ndarray:
    """Historical-only importance weights reflecting the two SHAP response directions."""
    absolute = np.abs(residual)
    typical = max(float(np.median(absolute)), 1e-6)
    if objective == "cost":
        return 1.0 + 2.0 * np.minimum(absolute / typical, 8.0)
    underforecast = np.clip(residual, 0.0, None)
    positive = underforecast[underforecast > 0]
    typical_under = max(float(np.median(positive)) if len(positive) else typical, 1e-6)
    return 1.0 + 3.0 * np.minimum(underforecast / typical_under, 8.0)


def weighted_ridge(
    x: np.ndarray,
    y: np.ndarray,
    base_weight: np.ndarray,
    age_days: np.ndarray,
    ridge: float,
    forgetting: float,
) -> np.ndarray:
    weights = base_weight * np.power(forgetting, np.maximum(age_days, 0.0))
    root = np.sqrt(np.maximum(weights, 1e-12))
    xw, yw = x * root[:, None], y * root
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = ridge * 0.05
    return np.linalg.solve(xw.T @ xw + penalty, xw.T @ yw)


def select_2021_gain(
    raw: pd.DataFrame,
    observed: pd.DataFrame,
    provinces: list[str],
    objective: str,
    args: argparse.Namespace,
) -> tuple[float, pd.DataFrame]:
    """Select a conservative correction gain on a nested 2021 time split."""
    origins = pd.DatetimeIndex(raw.forecast_origin_beijing.drop_duplicates().sort_values())
    split_origin = origins[max(2, int(len(origins) * 0.60))]
    gain_grid = np.asarray([0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0])
    losses = np.zeros(len(gain_grid), dtype=float)
    weight_total = 0.0
    for province in provinces:
        forecast = raw[province].to_numpy(float)
        truth = observed[province].to_numpy(float)
        residual = truth - forecast
        scale = max(float(np.quantile(truth, 0.995)), float(np.quantile(forecast, 0.995)), 1.0)
        lagged = pd.Series(residual).shift(24).rolling(24 * 14, min_periods=24).mean().fillna(0.0).to_numpy()
        x_parts = []
        for _, indices in raw.groupby("forecast_origin_beijing", sort=True).groups.items():
            index = np.asarray(list(indices), dtype=int)
            x_parts.append(design_matrix(
                forecast[index], pd.DatetimeIndex(raw.valid_time_beijing.iloc[index]),
                scale, float(lagged[index[0]]),
            ))
        x = np.vstack(x_parts)
        train_mask = raw.valid_time_beijing.to_numpy(dtype="datetime64[ns]") < split_origin.to_datetime64()
        validation_mask = ~train_mask
        train_residual = residual[train_mask]
        ages = (split_origin.to_datetime64() - raw.valid_time_beijing.to_numpy(dtype="datetime64[ns]")[train_mask]) / np.timedelta64(1, "D")
        theta = weighted_ridge(
            x[train_mask], train_residual / scale,
            objective_weights(train_residual, objective), np.asarray(ages, float),
            args.ridge, args.forgetting,
        )
        proposed = (x[validation_mask] @ theta) * scale
        validation_residual = residual[validation_mask]
        validation_weight = objective_weights(validation_residual, objective)
        for index, gain in enumerate(gain_grid):
            remaining = validation_residual - gain * proposed
            losses[index] += float(np.sum(validation_weight * np.abs(remaining)))
        weight_total += float(validation_weight.sum())
    mean_losses = losses / max(weight_total, 1e-12)
    best_index = int(np.argmin(mean_losses))
    table = pd.DataFrame({
        "objective": objective, "gain": gain_grid, "weighted_mae_mw": mean_losses,
        "selected": np.arange(len(gain_grid)) == best_index,
    })
    return float(gain_grid[best_index]), table


def calibrate_model(
    model_name: str, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecast_dir = OUT / "forecasts"
    raw_2021 = pd.read_csv(forecast_dir / f"{model_name}_2021_raw.csv", parse_dates=["forecast_origin_beijing", "valid_time_beijing"])
    raw_2022 = pd.read_csv(forecast_dir / f"{model_name}_2022_raw.csv", parse_dates=["forecast_origin_beijing", "valid_time_beijing"])
    obs_2021 = pd.read_csv(forecast_dir / "observed_2021.csv", parse_dates=["forecast_origin_beijing", "valid_time_beijing"])
    obs_2022 = pd.read_csv(forecast_dir / "observed_2022.csv", parse_dates=["forecast_origin_beijing", "valid_time_beijing"])
    meta = {"forecast_origin_beijing", "lead_hour", "valid_time_beijing"}
    provinces = [column for column in raw_2022.columns if column not in meta]
    outputs = []
    coefficients = []
    audits = []
    gain_tables = []
    for objective in OBJECTIVES:
        gain, gain_table = select_2021_gain(raw_2021, obs_2021, provinces, objective, args)
        gain_table.insert(0, "model", DISPLAY[model_name])
        gain_tables.append(gain_table)
        print(f"  {objective}: 2021-selected gain={gain:.2f}", flush=True)
        adjusted = raw_2022.copy()
        for province in provinces:
            train = pd.DataFrame({
                "origin": raw_2021.forecast_origin_beijing,
                "valid": raw_2021.valid_time_beijing,
                "forecast": raw_2021[province].to_numpy(float),
                "observed": obs_2021[province].to_numpy(float),
            })
            test = pd.DataFrame({
                "origin": raw_2022.forecast_origin_beijing,
                "valid": raw_2022.valid_time_beijing,
                "forecast": raw_2022[province].to_numpy(float),
                "observed": obs_2022[province].to_numpy(float),
            })
            scale = max(float(train.observed.quantile(0.995)), float(test.forecast.quantile(0.995)), 1.0)
            train_residual = (train.observed - train.forecast).to_numpy(float)
            train_lag = pd.Series(train_residual).shift(24).rolling(24 * 14, min_periods=24).mean().fillna(0.0).to_numpy()
            train_x_parts = []
            for _, indices in train.groupby("origin", sort=True).groups.items():
                index = np.asarray(list(indices), dtype=int)
                train_x_parts.append(design_matrix(
                    train.forecast.to_numpy(float)[index],
                    pd.DatetimeIndex(train.valid.iloc[index]), scale, float(train_lag[index[0]]),
                ))
            history_x = np.vstack(train_x_parts)
            history_residual = train_residual.copy()
            history_valid = train.valid.to_numpy(dtype="datetime64[ns]")
            corrected = np.empty(len(test), dtype=float)
            origins = pd.DatetimeIndex(test.origin.drop_duplicates().sort_values())
            for origin in origins:
                origin64 = origin.to_datetime64()
                cutoff64 = (origin - pd.Timedelta(days=args.window_days)).to_datetime64()
                mask = (history_valid <= origin64) & (history_valid >= cutoff64)
                available_indices = np.flatnonzero(mask)
                if len(available_indices) < 48:
                    available_indices = np.flatnonzero(history_valid <= origin64)[-24 * args.window_days:]
                residual_available = history_residual[available_indices]
                lagged_bias = float(residual_available[-24 * 14:].mean()) if len(residual_available) else 0.0
                x = history_x[available_indices]
                y = residual_available / scale
                weights = objective_weights(residual_available, objective)
                ages = (origin64 - history_valid[available_indices]) / np.timedelta64(1, "D")
                theta = weighted_ridge(x, y, weights, np.asarray(ages, float), args.ridge, args.forgetting)
                current_indices = np.flatnonzero(test.origin.to_numpy(dtype="datetime64[ns]") == origin64)
                current = test.iloc[current_indices]
                x_now = design_matrix(current.forecast.to_numpy(), pd.DatetimeIndex(current.valid), scale, lagged_bias)
                correction = gain * np.clip((x_now @ theta) * scale, -args.max_correction_share * scale, args.max_correction_share * scale)
                corrected[current_indices] = np.clip(current.forecast.to_numpy() + correction, 0.0, scale * 1.05)
                coefficients.extend({
                    "model": DISPLAY[model_name], "objective": objective, "province": province,
                    "forecast_origin_beijing": origin, "feature": feature, "coefficient": value,
                } for feature, value in zip(FEATURES, theta))
                audits.append({
                    "model": DISPLAY[model_name], "objective": objective, "province": province,
                    "forecast_origin_beijing": origin,
                    "latest_observation_used": pd.Timestamp(history_valid[available_indices].max()),
                    "training_rows": len(available_indices),
                    "leakage_free": bool(history_valid[available_indices].max() <= origin64),
                })
                # At the next daily origin, all 24 outcomes from today's forecast become available.
                day_residual = (current.observed - current.forecast).to_numpy(float)
                history_x = np.vstack([history_x, x_now])
                history_residual = np.r_[history_residual, day_residual]
                history_valid = np.r_[history_valid, current.valid.to_numpy(dtype="datetime64[ns]")]
            adjusted[province] = corrected
        adjusted.insert(0, "objective", objective)
        adjusted.insert(0, "model", DISPLAY[model_name])
        outputs.append(adjusted)
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(coefficients), pd.DataFrame(audits), pd.concat(gain_tables, ignore_index=True)


def evaluate_forecasts(adjusted: pd.DataFrame) -> pd.DataFrame:
    observed = pd.read_csv(OUT / "forecasts" / "observed_2022.csv", parse_dates=["valid_time_beijing"])
    meta = {"forecast_origin_beijing", "lead_hour", "valid_time_beijing"}
    provinces = [column for column in observed.columns if column not in meta]
    truth = observed[provinces].to_numpy(float)
    rows = []
    for model_name in MODELS:
        raw = pd.read_csv(OUT / "forecasts" / f"{model_name}_2022_raw.csv")[provinces].to_numpy(float)
        for objective in OBJECTIVES:
            calibrated = adjusted.loc[(adjusted.model == DISPLAY[model_name]) & (adjusted.objective == objective), provinces].to_numpy(float)
            for variant, prediction in (("Raw", raw), ("Dynamic linear", calibrated)):
                error = prediction - truth
                rows.append({
                    "model": DISPLAY[model_name], "objective": objective, "variant": variant,
                    "mae_mw": np.mean(np.abs(error)), "rmse_mw": np.sqrt(np.mean(error ** 2)),
                    "mean_bias_mw": np.mean(error),
                    "national_mae_gw": np.mean(np.abs(error.sum(axis=1))) / 1000.0,
                })
    metrics = pd.DataFrame(rows)
    raw = metrics.loc[metrics.variant == "Raw"].set_index(["model", "objective"])
    dynamic = metrics.loc[metrics.variant == "Dynamic linear"].set_index(["model", "objective"])
    comparison = dynamic.copy()
    for metric in ("mae_mw", "rmse_mw", "national_mae_gw"):
        comparison[f"{metric}_change_pct"] = 100 * (dynamic[metric] / raw[metric] - 1)
    comparison = comparison.reset_index()
    comparison["variant"] = "Dynamic linear vs Raw"
    return pd.concat([metrics, comparison], ignore_index=True, sort=False)


def run_calibration(args: argparse.Namespace) -> None:
    all_adjusted, all_coefficients, all_audits, all_gains = [], [], [], []
    for model_name in MODELS:
        print(f"dynamic calibration: {DISPLAY[model_name]}", flush=True)
        adjusted, coefficients, audit, gains = calibrate_model(model_name, args)
        all_adjusted.append(adjusted)
        all_coefficients.append(coefficients)
        all_audits.append(audit)
        all_gains.append(gains)
    adjusted = pd.concat(all_adjusted, ignore_index=True)
    coefficients = pd.concat(all_coefficients, ignore_index=True)
    audit = pd.concat(all_audits, ignore_index=True)
    gains = pd.concat(all_gains, ignore_index=True)
    adjusted.to_csv(OUT / "2022_dynamic_linear_forecasts.csv", index=False, float_format="%.4f")
    coefficients.to_csv(OUT / "2022_dynamic_linear_coefficients.csv", index=False, float_format="%.8f")
    audit.to_csv(OUT / "2022_information_audit.csv", index=False)
    gains.to_csv(OUT / "2021_gain_selection.csv", index=False, float_format="%.8f")
    metrics = evaluate_forecasts(adjusted)
    metrics.to_csv(OUT / "2022_forecast_metrics.csv", index=False, float_format="%.8f")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_training": "2021 chronological 75/25 train-validation split",
        "calibrator_initialisation": "final-quarter 2021 validation residuals only",
        "test_period": "2022 full year",
        "update_frequency": "daily at forecast origin",
        "observation_rule": "valid_time <= forecast_origin",
        "objectives": {
            "cost": "weighted toward large absolute historical errors",
            "co2": "weighted toward historical underforecast residuals",
        },
        "features": list(FEATURES), "window_days": args.window_days,
        "forgetting_factor": args.forgetting, "ridge": args.ridge,
        "max_correction_share": args.max_correction_share,
        "gain_selection": "nested chronological split within the 2021 validation quarter; zero gain allowed",
        "hyperparameter_selection": "fixed before 2022; no 2022 outcome-based candidate selection",
        "all_audit_rows_leakage_free": bool(audit.leakage_free.all()),
        "plot_created": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(metrics.loc[metrics.variant == "Dynamic linear vs Raw"].to_string(index=False), flush=True)


def load_dispatch_module():
    spec = importlib.util.spec_from_file_location("dynamic_linear_dispatch", DISPATCH_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {DISPATCH_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dispatch_inputs(operation) -> dict:
    adjusted = pd.read_csv(
        OUT / "2022_dynamic_linear_forecasts.csv",
        parse_dates=["forecast_origin_beijing", "valid_time_beijing"],
    )
    raw_frames = {
        model: pd.read_csv(
            OUT / "forecasts" / f"{model.lower()}_2022_raw.csv",
            parse_dates=["forecast_origin_beijing", "valid_time_beijing"],
        )
        for model in DISPLAY.values()
    }
    meta = {"model", "objective", "forecast_origin_beijing", "lead_hour", "valid_time_beijing"}
    provinces = [column for column in adjusted.columns if column not in meta]
    times = pd.DatetimeIndex(sorted(adjusted.valid_time_beijing.unique()))
    wind = operation.power("wind").reindex(times)[provinces]
    solar = operation.power("solar").reindex(times)[provinces]
    load = operation.power("load").reindex(times)[provinces]
    if wind.isna().any().any() or solar.isna().any().any() or load.isna().any().any():
        raise ValueError("2022 operational inputs contain missing values")
    scenarios = {}
    for display_name, frame in raw_frames.items():
        scenarios[f"{display_name}::Raw"] = frame.set_index("valid_time_beijing").reindex(times)[provinces].clip(lower=0.0)
        for objective in OBJECTIVES:
            block = adjusted.loc[(adjusted.model == display_name) & (adjusted.objective == objective)]
            scenarios[f"{display_name}::{objective}"] = block.set_index("valid_time_beijing").reindex(times)[provinces].clip(lower=0.0)
    caps = operation.capacities(provinces)
    for province in provinces:
        caps.at[province, "wind"] = max(
            caps.at[province, "wind"], 1.001 * wind[province].max(),
            *(1.001 * frame[province].max() for frame in scenarios.values()),
        )
        caps.at[province, "solar"] = max(caps.at[province, "solar"], 1.001 * solar[province].max())
    edges = pd.read_csv(
        operation.P / "resources/data/grids/edges_current.csv", header=None,
        names=["bus0", "bus1", "p_nom"],
    )
    edges = edges.query("bus0 in @provinces and bus1 in @provinces").reset_index(drop=True)
    return dict(
        times=times, provinces=provinces, wind=wind, solar=solar, load=load,
        scenarios=scenarios, caps=caps, edges=edges,
    )


def run_dispatch(args: argparse.Namespace) -> None:
    """Run objective-specific forecasts through the unchanged two-stage dispatch."""
    logging.getLogger("pypsa").setLevel(logging.ERROR)
    logging.getLogger("linopy").setLevel(logging.ERROR)
    operation = load_dispatch_module()
    data = dispatch_inputs(operation)
    factors = pd.read_csv(operation.FACTOR / "carrier_operational_co2_factors.csv").set_index("carrier").factor_t_co2_per_mwh
    cache_dir = OUT / "dispatch_cache"
    cache_dir.mkdir(exist_ok=True)
    periods = [
        (str(period), pd.DatetimeIndex(group.to_numpy()))
        for period, group in pd.Series(data["times"], index=data["times"]).groupby(data["times"].to_period("M"))
    ]
    annual_rows = []
    daily_frames = []
    scenarios = list(data["scenarios"])
    if args.dispatch_model:
        scenarios = [scenario for scenario in scenarios if scenario.startswith(f"{args.dispatch_model}::")]
    for scenario in scenarios:
        safe = scenario.replace("::", "_").lower()
        frames = []
        for period, times in periods:
            cache_path = cache_dir / f"{safe}_{period}.csv"
            if cache_path.exists() and not args.force_dispatch:
                hourly = pd.read_csv(cache_path, parse_dates=["timestamp_beijing"])
                print(f"cache {scenario:20s} {period}", flush=True)
            else:
                print(f"solve {scenario:20s} {period}: {len(times)} h", flush=True)
                hourly = operation.chunk(scenario, times, data, factors)
                hourly.to_csv(cache_path, index=False, float_format="%.6f")
            frames.append(hourly)
        hourly = pd.concat(frames, ignore_index=True)
        model, variant = scenario.split("::")
        hourly["model"] = model
        hourly["variant"] = variant
        hourly["date"] = pd.to_datetime(hourly.timestamp_beijing).dt.date
        daily = hourly.groupby(["model", "variant", "date"], as_index=False).agg(
            balancing_cost_million_cny=("balancing_cost_cny", lambda values: values.sum() / 1e6),
            fuel_cost_million_cny=("realised_fuel_cost_cny", lambda values: values.sum() / 1e6),
            total_operating_cost_million_cny=("total_operating_cost_cny", lambda values: values.sum() / 1e6),
            direct_co2_mt=("direct_co2_t", lambda values: values.sum() / 1e6),
            load_shedding_mwh=("load_shedding_mw", "sum"),
        )
        daily_frames.append(daily)
        annual_rows.append({
            "model": model, "variant": variant, "hours": len(hourly),
            "balancing_cost_million_cny": hourly.balancing_cost_cny.sum() / 1e6,
            "fuel_cost_million_cny": hourly.realised_fuel_cost_cny.sum() / 1e6,
            "total_operating_cost_million_cny": hourly.total_operating_cost_cny.sum() / 1e6,
            "direct_co2_mt": hourly.direct_co2_t.sum() / 1e6,
            "load_shedding_gwh": hourly.load_shedding_mw.sum() / 1e3,
        })
    annual = pd.DataFrame(annual_rows)
    raw = annual.loc[annual.variant == "Raw"].set_index("model")
    for index, row in annual.iterrows():
        reference = raw.loc[row.model]
        annual.loc[index, "balancing_cost_change_pct"] = 100 * (row.balancing_cost_million_cny / reference.balancing_cost_million_cny - 1)
        annual.loc[index, "total_operating_cost_change_pct"] = 100 * (
            row.total_operating_cost_million_cny / reference.total_operating_cost_million_cny - 1
        )
        annual.loc[index, "direct_co2_change_pct"] = 100 * (row.direct_co2_mt / reference.direct_co2_mt - 1)
    suffix = f"_{args.dispatch_model.lower()}" if args.dispatch_model else ""
    annual.to_csv(OUT / f"2022_dispatch_annual{suffix}.csv", index=False, float_format="%.8f")
    pd.concat(daily_frames, ignore_index=True).to_csv(OUT / f"2022_dispatch_daily{suffix}.csv", index=False, float_format="%.8f")
    print(annual.to_string(index=False), flush=True)


def summarize_dispatch() -> None:
    annual = pd.concat(
        [pd.read_csv(OUT / f"2022_dispatch_annual_{model.lower()}.csv") for model in DISPLAY.values()],
        ignore_index=True,
    )
    daily = pd.concat(
        [pd.read_csv(OUT / f"2022_dispatch_daily_{model.lower()}.csv") for model in DISPLAY.values()],
        ignore_index=True,
    )
    raw = annual.loc[annual.variant == "Raw"].set_index("model")
    for index, row in annual.iterrows():
        reference = raw.loc[row.model]
        annual.loc[index, "balancing_cost_change_pct"] = 100 * (row.balancing_cost_million_cny / reference.balancing_cost_million_cny - 1)
        annual.loc[index, "total_operating_cost_change_pct"] = 100 * (
            row.total_operating_cost_million_cny / reference.total_operating_cost_million_cny - 1
        )
        annual.loc[index, "direct_co2_change_pct"] = 100 * (row.direct_co2_mt / reference.direct_co2_mt - 1)
    annual.to_csv(OUT / "2022_dispatch_annual.csv", index=False, float_format="%.8f")
    daily.to_csv(OUT / "2022_dispatch_daily.csv", index=False, float_format="%.8f")
    rows = []
    for model in DISPLAY.values():
        reference = raw.loc[model]
        cost = annual.loc[(annual.model == model) & (annual.variant == "cost")].iloc[0]
        co2 = annual.loc[(annual.model == model) & (annual.variant == "co2")].iloc[0]
        rows.extend([
            {
                "model": model, "objective": "Total operating cost", "unit": "million CNY",
                "raw_value": reference.total_operating_cost_million_cny,
                "dynamic_value": cost.total_operating_cost_million_cny,
                "change_pct": cost.total_operating_cost_change_pct,
                "decreased": bool(cost.total_operating_cost_change_pct < 0),
            },
            {
                "model": model, "objective": "Direct CO2", "unit": "Mt",
                "raw_value": reference.direct_co2_mt,
                "dynamic_value": co2.direct_co2_mt,
                "change_pct": co2.direct_co2_change_pct,
                "decreased": bool(co2.direct_co2_change_pct < 0),
            },
        ])
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT / "2022_objective_specific_summary.csv", index=False, float_format="%.8f")
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "operational_validation": "full 8,760-hour two-stage PyPSA dispatch",
        "dispatch_outputs": ["2022_dispatch_daily.csv", "2022_dispatch_annual.csv"],
        "objective_summary": "2022_objective_specific_summary.csv",
        "cost_definition": "total operating cost = realised fuel cost + balancing cost",
        "cost_decreased_for_all_models": bool(comparison.loc[comparison.objective == "Total operating cost", "decreased"].all()),
        "co2_decreased_for_all_models": bool(comparison.loc[comparison.objective == "Direct CO2", "decreased"].all()),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "train", "calibrate", "dispatch", "summarize-dispatch"), default="all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--issue-hour", type=int, default=8)
    parser.add_argument("--window-days", type=int, default=60)
    parser.add_argument("--forgetting", type=float, default=0.98)
    parser.add_argument("--ridge", type=float, default=2.0)
    parser.add_argument("--max-correction-share", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-dispatch", action="store_true")
    parser.add_argument("--dispatch-model", choices=tuple(DISPLAY.values()))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.stage in ("all", "train"):
        train_and_predict(arguments)
    if arguments.stage in ("all", "calibrate"):
        run_calibration(arguments)
    if arguments.stage in ("all", "dispatch"):
        run_dispatch(arguments)
    if arguments.stage == "summarize-dispatch":
        summarize_dispatch()

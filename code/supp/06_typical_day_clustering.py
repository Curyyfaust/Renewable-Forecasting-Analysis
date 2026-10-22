"""Diagnostics for the representative-day clustering used in Figure 6.

The same 24 weighted seasonal k-medoids are used for the 2020 storage
sensitivity and the controlled 2030/2050 SSP1.5 states. The figure shows
coverage of the 365 daily states, cluster weights and compactness, and
preservation of the eight clustering features after representative-day
weighting.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "06_sen"
OUTPUT = ROOT / "supp" / "supp_figure"
OPERATION = ROOT / "PyPSA-china" / "main_dispatch.py"
REPRESENTATIVES = RESULTS / "06_representative_days.csv"

SEASONS = ("Winter", "Spring", "Summer", "Autumn")
SEASON_FOR_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}
FEATURES = (
    "mean_load", "peak_load", "load_sd", "mean_wind",
    "wind_sd", "peak_wind", "mean_solar", "peak_solar",
)
FEATURE_LABELS = (
    "Mean load", "Peak load", "Load SD", "Mean wind",
    "Wind SD", "Peak wind", "Mean solar", "Peak solar",
)


def load_operation():
    spec = importlib.util.spec_from_file_location("ssp15_dispatch", OPERATION)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def system_time_series() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_operation().inputs()
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
    daily = daily.loc[hourly.groupby("date").size().eq(24)].copy()
    daily["season"] = daily.index.month.map(SEASON_FOR_MONTH)
    return daily, hourly.drop(columns="date")


def assign_clusters(daily: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    assigned = []
    for season in SEASONS:
        block = daily.loc[daily.season.eq(season)].copy()
        medoids = selected.loc[selected.season.eq(season)].sort_values("cluster")
        scale = (
            block[list(FEATURES)].quantile(0.75)
            - block[list(FEATURES)].quantile(0.25)
        ).replace(0, 1)
        centre = block[list(FEATURES)].median()
        values = ((block[list(FEATURES)] - centre) / scale).to_numpy(float)
        medoid_values = (
            (block.loc[pd.DatetimeIndex(medoids.representative_date), list(FEATURES)] - centre)
            / scale
        ).to_numpy(float)
        distance = np.sqrt(((values[:, None, :] - medoid_values[None, :, :]) ** 2).sum(axis=2))
        labels = distance.argmin(axis=1)
        block["cluster"] = labels + 1
        block["distance_to_medoid"] = distance[np.arange(len(block)), labels]
        assigned.append(block)
    return pd.concat(assigned).sort_index()


def pca_coordinates(daily: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = daily[list(FEATURES)].copy()
    scale = (values.quantile(0.75) - values.quantile(0.25)).replace(0, 1)
    standardized = ((values - values.median()) / scale).to_numpy(float)
    _, singular, right = np.linalg.svd(standardized, full_matrices=False)
    coordinates = standardized @ right[:2].T
    explained = singular[:2] ** 2 / np.sum(singular ** 2) * 100
    return coordinates, explained


def build_audit_table(daily: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    records = []
    for season in SEASONS:
        season_days = daily.loc[daily.season.eq(season)]
        medoids = selected.loc[selected.season.eq(season)].sort_values("cluster")
        for row in medoids.itertuples(index=False):
            members = season_days.loc[season_days.cluster.eq(row.cluster)]
            records.append(
                {
                    "record_type": "cluster",
                    "season": season,
                    "cluster": int(row.cluster),
                    "representative_date": pd.Timestamp(row.representative_date).date().isoformat(),
                    "weight_days": int(row.weight_days),
                    "assigned_days": int(len(members)),
                    "mean_distance": float(members.distance_to_medoid.mean()),
                    "max_distance": float(members.distance_to_medoid.max()),
                }
            )
    weighted = daily.loc[pd.DatetimeIndex(selected.representative_date), list(FEATURES)].copy()
    weights = selected.set_index("representative_date").loc[weighted.index, "weight_days"].to_numpy()
    for feature, label in zip(FEATURES, FEATURE_LABELS):
        annual = float(daily[feature].mean())
        representative = float(np.average(weighted[feature], weights=weights))
        records.append(
            {
                "record_type": "feature",
                "feature": feature,
                "feature_label": label,
                "full_year_mean": annual,
                "weighted_representative_mean": representative,
                "relative_error_pct": 100 * (representative - annual) / annual,
            }
        )
    return pd.DataFrame(records)


def make_figure(
    daily: pd.DataFrame,
    hourly: pd.DataFrame,
    selected: pd.DataFrame,
    audit: pd.DataFrame,
) -> plt.Figure:
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(15.5, 13.2))
    grid = fig.add_gridspec(3, 2, height_ratios=(1, 1, 0.48))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    time_grid = grid[2, :].subgridspec(1, 3, wspace=0.28)
    time_axes = [fig.add_subplot(time_grid[0, index]) for index in range(3)]
    colors = {
        "Winter": "#548DCF", "Spring": "#59C3BE",
        "Summer": "#F6D66F", "Autumn": "#E98A65",
    }

    coordinates, explained = pca_coordinates(daily)
    for season in SEASONS:
        mask = daily.season.eq(season).to_numpy()
        ax_a.scatter(coordinates[mask, 0], coordinates[mask, 1], s=24,
                     alpha=0.42, color=colors[season], label=season)
        medoid_dates = pd.DatetimeIndex(
            selected.loc[selected.season.eq(season), "representative_date"]
        )
        positions = daily.index.get_indexer(medoid_dates)
        weights = selected.loc[selected.season.eq(season), "weight_days"].to_numpy()
        ax_a.scatter(coordinates[positions, 0], coordinates[positions, 1],
                     s=90 + 8 * weights, marker="*", color=colors[season],
                     edgecolor="black", linewidth=0.8, zorder=4)
    medoid_handle = Line2D([0], [0], marker="*", linestyle="none", markersize=14,
                           markerfacecolor="0.75", markeredgecolor="black",
                           label="Medoid (size = weight)")
    ax_a.axhline(0, color="0.8", linewidth=0.8)
    ax_a.axvline(0, color="0.8", linewidth=0.8)
    ax_a.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
    ax_a.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
    ax_a.set_title("Coverage of daily system states")
    handles, labels = ax_a.get_legend_handles_labels()
    ax_a.legend(handles + [medoid_handle], labels + [medoid_handle.get_label()],
                frameon=True, framealpha=0.92, ncol=2, loc="upper right",
                fontsize=12)

    clusters = audit.loc[audit.record_type.eq("cluster")].copy()
    prefixes = {"Winter": "W", "Spring": "Sp", "Summer": "Su", "Autumn": "A"}
    clusters["label"] = clusters.apply(
        lambda row: f"{prefixes[row['season']]}{int(row['cluster'])}", axis=1
    )
    x = np.arange(len(clusters))
    for season in SEASONS:
        mask = clusters.season.eq(season).to_numpy()
        ax_b.bar(x[mask], clusters.loc[mask, "weight_days"],
                 color=colors[season], label=season)
    ax_b.set_xticks(x, clusters.label, rotation=45, ha="right")
    ax_b.tick_params(axis="x", labelsize=10)
    ax_b.set_ylabel("Days represented")
    ax_b.set_xlabel("Seasonal cluster")
    ax_b.set_title("Cluster weights")
    ax_b.text(0.98, 0.95, f"Total = {int(clusters.weight_days.sum())} days",
              transform=ax_b.transAxes, ha="right", va="top")

    for season in SEASONS:
        block = clusters.loc[clusters.season.eq(season)]
        positions = block.index.to_numpy()
        ax_c.vlines(positions, block.mean_distance, block.max_distance,
                    color=colors[season], linewidth=2)
        ax_c.scatter(positions, block.mean_distance, color=colors[season],
                     s=55, zorder=3)
        ax_c.scatter(positions, block.max_distance, facecolors="none",
                     edgecolors=colors[season], s=65, zorder=3)
    ax_c.set_xticks(x, clusters.label, rotation=45, ha="right")
    ax_c.tick_params(axis="x", labelsize=10)
    ax_c.set_xlabel("Seasonal cluster")
    ax_c.set_ylabel("Robust standardized distance")
    ax_c.set_title("Within-cluster compactness")
    ax_c.plot([], [], "o", color="0.25", label="Mean")
    ax_c.plot([], [], "o", markerfacecolor="none", markeredgecolor="0.25",
              label="Maximum")
    ax_c.legend(frameon=False, ncol=2, loc="upper left")

    features = audit.loc[audit.record_type.eq("feature")].copy()
    y = np.arange(len(features))
    values = features.relative_error_pct.to_numpy(float)
    ax_d.axvline(0, color="0.35", linewidth=1)
    ax_d.barh(y, values)
    ax_d.set_yticks(y, features.feature_label)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Weighted-medoid bias relative to full year (%)")
    ax_d.set_title("Preservation of annual daily statistics")
    padding = max(0.08 * np.max(np.abs(values)), 0.05)
    for yi, value in zip(y, values):
        if value < -1:
            ax_d.text(-0.15, yi, f"{value:+.2f}%", va="center",
                      ha="right", color="white")
        else:
            ax_d.text(value + np.sign(value or 1) * padding, yi, f"{value:+.2f}%",
                      va="center", ha="left" if value >= 0 else "right")
    limit = max(np.max(np.abs(values)) * 1.35, 0.5)
    ax_d.set_xlim(-limit, limit)

    series_spec = (("load", "Load"), ("wind", "Wind generation"),
                   ("solar", "Solar generation"))
    hours = np.arange(24)
    for ax, (variable, title) in zip(time_axes, series_spec):
        for season in SEASONS:
            medoids = selected.loc[selected.season.eq(season)].sort_values("cluster")
            profiles = []
            weights = []
            for row in medoids.itertuples(index=False):
                date = pd.Timestamp(row.representative_date)
                profile = hourly.loc[hourly.index.floor("D") == date, variable].to_numpy(float)
                if len(profile) != 24:
                    raise ValueError(f"Incomplete representative-day profile: {date}")
                profile = profile / 1000.0
                profiles.append(profile)
                weights.append(row.weight_days)
                ax.plot(hours, profile, color=colors[season], alpha=0.28,
                        linewidth=1.0)
            seasonal_mean = np.average(np.vstack(profiles), axis=0, weights=weights)
            ax.plot(hours, seasonal_mean, color=colors[season], linewidth=2.7,
                    label=season)
        ax.set_title(title)
        ax.set_xlabel("Hour")
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlim(0, 23)
    time_axes[0].set_ylabel("National total (GW)")
    time_axes[0].legend(frameon=False, ncol=2, fontsize=11,
                        columnspacing=1.0, handlelength=1.8)
    time_axes[1].text(
        0.5, 1.27,
        "Twenty-four-hour profiles of the selected cluster medoids",
        transform=time_axes[1].transAxes, ha="center", va="bottom",
        fontsize=17,
    )

    for label, ax in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes,
                fontsize=19, fontweight="bold")
    time_axes[0].text(-0.12, 1.20, "e", transform=time_axes[0].transAxes,
                      fontsize=19, fontweight="bold")
    fig.suptitle("Representative-day coverage for Figure 6",
                 fontsize=20, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.075, top=0.935,
                        wspace=0.28, hspace=0.56)
    return fig


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(REPRESENTATIVES, parse_dates=["representative_date"])
    daily, hourly = system_time_series()
    daily = assign_clusters(daily, selected)
    audit = build_audit_table(daily, selected)
    audit.to_csv(OUTPUT / "06_typical_day_clustering_values.csv", index=False)
    figure = make_figure(daily, hourly, selected, audit)
    figure.savefig(OUTPUT / "06_typical_day_clustering.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "06_typical_day_clustering.png", dpi=300,
                   bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT / "06_typical_day_clustering.pdf")


if __name__ == "__main__":
    main()

"""Seasonal balancing-response profiles supporting main-text Figure 6.

The four columns reproduce the storage states highlighted in Figure 6:
2020 without storage, 2020 with 20% / 4 h storage, the 2030 SSP1.5 state
with 10% / 4 h storage, and the 2050 SSP1.5 state with 20% / 4 h storage.
Rows show the largest-weight representative day in each season for CNN.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "06_sen" / "06_storage_future_hourly.csv.gz"
DAYS = ROOT / "results" / "06_sen" / "06_representative_days.csv"
OUTPUT = ROOT / "supp" / "supp_figure"

MODEL = "CNN"
SEASONS = ("Winter", "Spring", "Summer", "Autumn")
SCENARIOS = (
    ("2020_S00", "2020: no storage"),
    ("2020_S20", "2020: 20% / 4 h"),
    ("2030_S10", "2030: 10% / 4 h"),
    ("2050_S20", "2050: 20% / 4 h"),
)
UP_COLORS = ("#24677A", "#4EAAA7", "#F1AE55", "#9DD9E8")
DOWN_COLORS = ("#31864A", "#65B96E", "#F1AE55", "#C9DE8A")
UP_LABELS = ("Conventional up", "Renewables up", "Battery discharge", "Net imports")
DOWN_LABELS = ("Conventional down", "Renewables down", "Battery charge", "Curtailment")


def load_cases() -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly = pd.read_csv(INPUT, parse_dates=["timestamp", "representative_date"])
    days = pd.read_csv(DAYS, parse_dates=["representative_date"])
    selected = (
        days.sort_values(["season", "weight_days"], ascending=[True, False])
        .groupby("season", as_index=False, observed=True)
        .first()
    )
    selected["season"] = pd.Categorical(selected.season, SEASONS, ordered=True)
    selected = selected.sort_values("season")
    scenario_names = [scenario for scenario, _ in SCENARIOS]
    cases = hourly.loc[
        hourly.scenario.isin(scenario_names)
        & hourly.model.eq(MODEL)
        & hourly.representative_date.isin(selected.representative_date)
    ].copy()
    cases = cases.merge(
        selected[["season", "representative_date", "weight_days"]],
        on="representative_date", how="inner", validate="many_to_one",
        suffixes=("", "_selected"),
    )
    if "weight_days_selected" in cases:
        cases["weight_days"] = cases.pop("weight_days_selected")
    counts = cases.groupby(["scenario", "season"], observed=True).size()
    expected = pd.MultiIndex.from_product([scenario_names, SEASONS])
    counts = counts.reindex(expected, fill_value=0)
    if not (counts == 24).all():
        raise ValueError(f"Expected 24 hours for every scenario-season block: {counts.to_dict()}")
    return cases, selected


def balancing_components(block: pd.DataFrame) -> tuple[list[np.ndarray], list[np.ndarray]]:
    conventional = (
        block[[f"rt_{tech}_gw" for tech in ("hydro", "nuclear", "coal", "gas")]].to_numpy()
        - block[[f"da_{tech}_gw" for tech in ("hydro", "nuclear", "coal", "gas")]].to_numpy()
    )
    renewable = (
        block[["rt_wind_gw", "rt_solar_gw"]].to_numpy()
        - block[["da_wind_gw", "da_solar_gw"]].to_numpy()
    )
    upward = [
        np.clip(conventional, 0, None).sum(axis=1),
        np.clip(renewable, 0, None).sum(axis=1),
        np.clip(block.rt_battery_gw.to_numpy(float), 0, None),
        np.clip(block.link_net_change_gw.to_numpy(float), 0, None),
    ]
    downward = [
        np.clip(-conventional, 0, None).sum(axis=1),
        np.clip(-renewable, 0, None).sum(axis=1),
        np.clip(-block.rt_battery_gw.to_numpy(float), 0, None),
        np.clip(block.curtailment_gwh.to_numpy(float), 0, None),
    ]
    return upward, downward


def draw_signed_stack(
    ax: plt.Axes,
    hours: np.ndarray,
    upward: list[np.ndarray],
    downward: list[np.ndarray],
) -> None:
    ax.stackplot(hours, upward, colors=UP_COLORS, alpha=0.97, linewidth=0)
    baseline = np.zeros_like(hours, dtype=float)
    for values, color in zip(downward, DOWN_COLORS):
        next_baseline = baseline - values
        ax.fill_between(hours, baseline, next_baseline, color=color, alpha=0.97, linewidth=0)
        baseline = next_baseline
    ax.axhline(0, color="#5E6468", linewidth=0.8)
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18])
    ax.grid(axis="y", color="#D7DDE1", linewidth=0.6, linestyle=":", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)


def make_figure(cases: pd.DataFrame, selected: pd.DataFrame) -> plt.Figure:
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 13.5,
        "axes.titlesize": 15.5, "axes.titleweight": "bold",
        "axes.labelsize": 14.5, "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5, "legend.fontsize": 11.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(4, 4, figsize=(18.5, 14.8), sharex=True)
    panel_limits: dict[tuple[str, str], float] = {}
    for season in SEASONS:
        for scenario, _ in SCENARIOS:
            block = cases.loc[cases.scenario.eq(scenario) & cases.season.eq(season)]
            upward, downward = balancing_components(block.sort_values("timestamp"))
            panel_limits[(season, scenario)] = 1.08 * max(
                float(np.sum(upward, axis=0).max()),
                float(np.sum(downward, axis=0).max()),
            )
        present_limit = max(
            panel_limits[(season, "2020_S00")],
            panel_limits[(season, "2020_S20")],
        )
        panel_limits[(season, "2020_S00")] = present_limit
        panel_limits[(season, "2020_S20")] = present_limit

    dates = selected.set_index("season").representative_date
    panel_index = 0
    for row, season in enumerate(SEASONS):
        for column, (scenario, title) in enumerate(SCENARIOS):
            ax = axes[row, column]
            block = cases.loc[
                cases.scenario.eq(scenario) & cases.season.eq(season)
            ].sort_values("timestamp")
            upward, downward = balancing_components(block)
            draw_signed_stack(ax, block.timestamp.dt.hour.to_numpy(float), upward, downward)
            limit = panel_limits[(season, scenario)]
            ax.set_ylim(-limit, limit)
            if row == 0:
                ax.set_title(title, pad=7)
            if column == 0:
                date = pd.Timestamp(dates.loc[season]).strftime("%d %b")
                ax.set_ylabel(f"{season} ({date})\nBalancing response (GW)")
            if row == len(SEASONS) - 1:
                ax.set_xlabel("Hour")
            ax.text(0.015, 0.965, chr(ord("a") + panel_index), transform=ax.transAxes,
                    fontsize=16.5, fontweight="bold", ha="left", va="top")
            panel_index += 1

    legend_handles = [Patch(facecolor=color, label=label)
                      for color, label in zip(UP_COLORS, UP_LABELS)]
    legend_handles += [Patch(facecolor=color, label=label)
                       for color, label in zip(DOWN_COLORS, DOWN_LABELS)]
    legend_handles.append(Line2D([0], [0], color="#5E6468", lw=0.9, label="Zero response"))
    fig.legend(handles=legend_handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.995), columnspacing=1.2, handletextpad=0.45)
    fig.suptitle("Seasonal balancing responses across Figure 6 storage states (CNN)",
                 fontsize=19, fontweight="bold", y=0.955)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.065, top=0.90,
                        wspace=0.26, hspace=0.22)
    return fig


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases, selected = load_cases()
    cases.to_csv(OUTPUT / "06_typical_day_dispatch_values.csv", index=False)
    figure = make_figure(cases, selected)
    figure.savefig(OUTPUT / "06_typical_day_dispatch.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "06_typical_day_dispatch.png", dpi=260, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {OUTPUT / '06_typical_day_dispatch.pdf'}")


if __name__ == "__main__":
    main()

"""Supplementary counterfactual diagnostics for the Figure 05 experiment.

The script adds two views not shown explicitly in the main figure:
season-specific complete-removal effects and full daily effect distributions.
All graphical colours and elements use Matplotlib defaults; only font sizes
and layout are adjusted for readability.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "05_counterfactual_validation.csv"
OUTPUT = ROOT / "supp" / "supp_figure"

INTERVENTIONS = ("TB", "TV", "SB", "SV")
INTERVENTION_NAMES = {
    "TB": "Temporal bias",
    "TV": "Temporal variability",
    "SB": "Spatial bias",
    "SV": "Spatial variability",
}
SEASONS = ("Winter", "Spring", "Summer", "Autumn")


def load_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT)
    required = {"model", "day_id", "season", "intervention", "alpha", "target", "actual_delta"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Counterfactual table is missing: {sorted(missing)}")
    return data


def seasonal_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize complete-removal effects by season with 95% normal CIs."""
    removed = data.loc[np.isclose(data.alpha, 0.0)].copy()
    summary = (
        removed.groupby(["target", "season", "intervention"], observed=True)
        .actual_delta.agg(["mean", "std", "count"]).reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci95"] = 1.96 * summary["sem"]
    return summary


def distribution_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Store the statistics represented by the daily boxplots."""
    return (
        data.groupby(["target", "intervention", "alpha"], observed=True)
        .actual_delta.agg(
            samples="count", mean="mean", std="std", median="median",
            q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75),
            minimum="min", maximum="max",
        )
        .reset_index()
    )


def make_seasonal_figure(summary: pd.DataFrame) -> plt.Figure:
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 15})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    specs = (
        ("cost", "Balancing cost", r"$\Delta$ cost (M CNY day$^{-1}$)", 1.0),
        ("co2", r"CO$_2$ emissions", r"$\Delta$ CO$_2$ (kt day$^{-1}$)", 1000.0),
    )
    y = np.arange(len(INTERVENTIONS), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(SEASONS))

    for panel, (ax, (target, title, xlabel, multiplier)) in enumerate(zip(axes, specs)):
        subset = summary.loc[summary.target == target]
        for offset, season in zip(offsets, SEASONS):
            block = subset.loc[subset.season == season].set_index("intervention").loc[list(INTERVENTIONS)]
            ax.errorbar(
                block["mean"].to_numpy() * multiplier,
                y + offset,
                xerr=block.ci95.to_numpy() * multiplier,
                fmt="o",
                capsize=3,
                label=season,
            )
        ax.axvline(0)
        ax.set_yticks(y, [INTERVENTION_NAMES[key] for key in INTERVENTIONS])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=17)
        ax.set_title(title, fontsize=19)
        ax.tick_params(labelsize=14)
        ax.text(-0.12, 1.05, chr(ord("a") + panel), transform=ax.transAxes,
                fontsize=20, fontweight="bold")
    axes[1].tick_params(labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=15)
    fig.suptitle("Seasonal complete-removal effects", fontsize=20, y=0.91)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.16, top=0.76, wspace=0.20)
    return fig


def make_distribution_figure(data: pd.DataFrame) -> plt.Figure:
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 15})
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.8))
    specs = (
        ("cost", "Balancing cost", r"Daily $\Delta$ cost (M CNY day$^{-1}$)", 1.0),
        ("co2", r"CO$_2$ emissions", r"Daily $\Delta$ CO$_2$ (kt day$^{-1}$)", 1000.0),
    )
    positions = np.arange(1, 9)
    labels = []
    for intervention in INTERVENTIONS:
        labels.extend([f"{intervention}\n50%", f"{intervention}\nRemoved"])

    for panel, (ax, (target, title, ylabel, multiplier)) in enumerate(zip(axes, specs)):
        subset = data.loc[data.target == target]
        values = []
        for intervention in INTERVENTIONS:
            values.append(subset.loc[(subset.intervention == intervention) &
                                     np.isclose(subset.alpha, 0.5), "actual_delta"].to_numpy() * multiplier)
            values.append(subset.loc[(subset.intervention == intervention) &
                                     np.isclose(subset.alpha, 0.0), "actual_delta"].to_numpy() * multiplier)
        ax.boxplot(values, positions=positions, widths=0.65, showfliers=False)
        ax.axhline(0)
        ax.set_xticks(positions, labels)
        ax.set_ylabel(ylabel, fontsize=17)
        ax.set_title(title, fontsize=19)
        ax.tick_params(labelsize=13)
        ax.text(-0.10, 1.05, chr(ord("a") + panel), transform=ax.transAxes,
                fontsize=20, fontweight="bold")
    fig.suptitle("Distribution across models and representative days", fontsize=20, y=0.95)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.19, top=0.82, wspace=0.22)
    return fig


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.pdf")
    fig.savefig(OUTPUT / f"{stem}.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    seasonal = seasonal_summary(data)
    distributions = distribution_summary(data)
    seasonal.to_csv(OUTPUT / "05_seasonal_effects_values.csv", index=False)
    distributions.to_csv(OUTPUT / "05_daily_effect_distributions_values.csv", index=False)

    save_figure(make_seasonal_figure(seasonal), "05_seasonal_effects")
    save_figure(make_distribution_figure(data), "05_daily_effect_distributions")
    print(f"Saved {OUTPUT / '05_seasonal_effects.pdf'}")
    print(f"Saved {OUTPUT / '05_daily_effect_distributions.pdf'}")


if __name__ == "__main__":
    main()

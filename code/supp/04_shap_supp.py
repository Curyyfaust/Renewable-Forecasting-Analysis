"""Supplementary SHAP contribution diagnostics.

Two figures complement the pooled SHAP analysis in the main Figure 04:

* 04_model_contributions: contribution structure for each forecast model.
* 04_seasonal_contributions: contribution structure for each season.

Both figures aggregate the already-computed sample-level SHAP values; no
model is refitted.  Plot colours and graphical elements use Matplotlib's
defaults, with only font sizes and layout adjusted for readability.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "supp" / "supp_figure"

MODELS = ("CNN", "LSTM", "Transformer", "DNN")
SEASONS = ("Winter", "Spring", "Summer", "Autumn")
GROUPS = ("Temporal bias", "Temporal variability", "Spatial bias", "Spatial variability")
METADATA = {"model", "day_id", "season", "date"}


def feature_group(name: str) -> str:
    """Map a feature name to the same four groups used in the main figure."""
    if name in {"underforecast_bias_gw", "overforecast_bias_gw"}:
        return "Temporal bias"
    if name in {
        "maximum_hourly_error_gw", "temporal_error_sd_gw",
        "ramp_error_gw", "error_persistence_h",
    }:
        return "Temporal variability"
    if name == "provincial_abs_bias_gw" or name.endswith("_under_gw") or name.endswith("_over_gw"):
        return "Spatial bias"
    return "Spatial variability"


def aggregate_shap() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate absolute SHAP contributions by model and by season."""
    model_rows: list[dict[str, float | str]] = []
    season_rows: list[dict[str, float | str]] = []
    for target in ("cost", "co2"):
        frame = pd.read_csv(RESULTS / f"04_shap_values_{target}.csv")
        features = [column for column in frame.columns if column not in METADATA]
        grouped_features = {
            group: [feature for feature in features if feature_group(feature) == group]
            for group in GROUPS
        }
        if any(not columns for columns in grouped_features.values()):
            raise ValueError(f"At least one SHAP feature group is empty for {target}")

        for dimension, categories, rows in (
            ("model", MODELS, model_rows),
            ("season", SEASONS, season_rows),
        ):
            for category in categories:
                subset = frame.loc[frame[dimension] == category]
                contributions = {
                    group: float(subset[columns].abs().to_numpy().sum())
                    for group, columns in grouped_features.items()
                }
                total = sum(contributions.values())
                sample_count = len(subset)
                for group in GROUPS:
                    rows.append({
                        "target": target,
                        dimension: category,
                        "group": group,
                        "samples": sample_count,
                        "mean_abs_shap": contributions[group] / sample_count,
                        "share_pct": 100.0 * contributions[group] / total,
                        "total_mean_abs_shap": total / sample_count,
                    })
    return pd.DataFrame(model_rows), pd.DataFrame(season_rows)


def plot_contribution_figure(
    data: pd.DataFrame,
    category_column: str,
    categories: tuple[str, ...],
    panel_name: str,
) -> plt.Figure:
    """Plot paired 100% stacked bars using Matplotlib's default colour cycle."""
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 15})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    target_specs = (
        ("cost", "Balancing cost", r"M CNY day$^{-1}$"),
        ("co2", r"CO$_2$ emissions", r"Mt day$^{-1}$"),
    )
    y = np.arange(len(categories))

    for panel_index, (ax, (target, title, unit)) in enumerate(zip(axes, target_specs)):
        subset = data.loc[data.target == target]
        left = np.zeros(len(categories))
        for group in GROUPS:
            values = np.array([
                subset.loc[(subset[category_column] == category) & (subset.group == group),
                           "share_pct"].iat[0]
                for category in categories
            ])
            ax.barh(y, values, left=left, label=group)
            for row, (start, value) in enumerate(zip(left, values)):
                if value >= 7.0:
                    ax.text(start + value / 2.0, row, f"{value:.0f}%",
                            ha="center", va="center", fontsize=13)
            left += values

        totals = np.array([
            subset.loc[subset[category_column] == category, "total_mean_abs_shap"].iat[0]
            for category in categories
        ])
        for row, total in enumerate(totals):
            label = f"{total:.2f}" if target == "cost" else f"{total:.3f}"
            ax.text(102.0, row, label, ha="left", va="center", fontsize=13)

        ax.set_yticks(y, categories)
        ax.invert_yaxis()
        ax.set_xlim(0, 120)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel(r"Contribution to mean $|\mathrm{SHAP}|$ (%)", fontsize=16)
        ax.set_title(title, fontsize=19)
        ax.tick_params(labelsize=14)
        ax.text(102.0, -0.62, f"Total\n({unit})", ha="left", va="bottom", fontsize=13)
        ax.text(-0.10, 1.05, chr(ord("a") + panel_index), transform=ax.transAxes,
                fontsize=20, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=14)
    fig.suptitle(panel_name, fontsize=20, y=0.91)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.76, wspace=0.28)
    return fig


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.pdf")
    fig.savefig(OUTPUT / f"{stem}.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_data, season_data = aggregate_shap()
    model_data.to_csv(OUTPUT / "04_model_contributions_values.csv", index=False)
    season_data.to_csv(OUTPUT / "04_seasonal_contributions_values.csv", index=False)

    model_figure = plot_contribution_figure(
        model_data, "model", MODELS, "Contribution structure across forecast models"
    )
    save_figure(model_figure, "04_model_contributions")

    seasonal_figure = plot_contribution_figure(
        season_data, "season", SEASONS, "Seasonal stability of contribution structure"
    )
    save_figure(seasonal_figure, "04_seasonal_contributions")

    for stem in ("04_model_contributions", "04_seasonal_contributions"):
        print(f"Saved {OUTPUT / (stem + '.pdf')}")
        print(f"Saved {OUTPUT / (stem + '.png')}")


if __name__ == "__main__":
    main()

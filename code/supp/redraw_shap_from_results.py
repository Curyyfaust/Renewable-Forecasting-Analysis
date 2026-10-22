"""Redraw the main SHAP figure from saved results without refitting models."""
from pathlib import Path
import importlib.util

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('main_shap_figure', ROOT / '04_shap.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
results = ROOT / 'results'
frame = pd.read_csv(results / '04_shap_features.csv')
columns = module.feature_columns(frame)
explained = {}
for target in ('total_cost', 'balancing_cost', 'co2'):
    saved = pd.read_csv(results / f'04_shap_values_{target}.csv')
    assert saved[['model', 'day_id']].equals(frame[['model', 'day_id']])
    explained[target] = {
        'features': frame[columns],
        'columns': columns,
        'values': saved[columns].to_numpy(),
    }
png, pdf = module.make_figure(
    pd.read_csv(results / '04_shap_validation.csv'),
    explained,
    pd.read_csv(results / '04_shap_grouped_contributions.csv'),
    pd.read_csv(results / '04_shap_region_season.csv'),
    ROOT / 'figure',
)
print(png)
print(pdf)

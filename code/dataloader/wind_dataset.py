"""PyTorch dataset for 24-hour provincial wind-power forecasting.

Each sample uses the previous 72 hourly wind-power observations and the next
24 hourly weather forecasts to predict the next 24 hourly wind-power values.
The default training period is strictly limited to calendar year 2019.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WIND_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "provincial_hourly_2018_2023"
    / "provincial_hourly_wind_2018_2023.csv"
)
DEFAULT_WEATHER_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "weather_forecast_24h"
    / "weatherbench2_hres_china_2018_2022.npz"
)


class WindForecastDataset(Dataset):
    """Align wind-power history, future weather, and future wind targets.

    The forecast origin is the final timestamp included in ``history_wind``.
    Forecast lead 1 corresponds to one hour after the origin. Samples are kept
    only when the complete 72-hour history and 24-hour target are inside the
    requested calendar year.
    """

    history_hours = 72
    forecast_hours = 24

    def __init__(
        self,
        wind_path: str | Path = DEFAULT_WIND_PATH,
        weather_path: str | Path = DEFAULT_WEATHER_PATH,
        year: int = 2019,
    ) -> None:
        self.wind_path = Path(wind_path)
        self.weather_path = Path(weather_path)
        self.year = int(year)

        wind_frame = pd.read_csv(self.wind_path, parse_dates=["timestamp_beijing"])
        if wind_frame["timestamp_beijing"].duplicated().any():
            raise ValueError("Wind data contains duplicate timestamps")
        wind_frame = wind_frame.set_index("timestamp_beijing").sort_index()

        weather = np.load(self.weather_path, allow_pickle=False)
        required_keys = {"origin", "provinces", "features", "forecast"}
        missing = required_keys.difference(weather.files)
        if missing:
            raise ValueError(f"Weather archive is missing keys: {sorted(missing)}")

        self.origins_all = pd.DatetimeIndex(weather["origin"])
        self.provinces = [str(value) for value in weather["provinces"].tolist()]
        self.weather_features = [str(value) for value in weather["features"].tolist()]
        forecast = weather["forecast"]

        expected_shape = (
            len(self.origins_all),
            self.forecast_hours,
            len(self.provinces),
            len(self.weather_features),
        )
        if forecast.shape != expected_shape:
            raise ValueError(
                f"Unexpected forecast shape {forecast.shape}; expected {expected_shape}"
            )
        missing_provinces = [p for p in self.provinces if p not in wind_frame.columns]
        if missing_provinces:
            raise ValueError(f"Wind data is missing provinces: {missing_provinces}")

        year_start = pd.Timestamp(year=self.year, month=1, day=1)
        year_end = pd.Timestamp(year=self.year + 1, month=1, day=1)
        first_valid_origin = year_start + pd.Timedelta(hours=self.history_hours - 1)
        last_valid_origin = year_end - pd.Timedelta(hours=self.forecast_hours + 1)
        selected = np.flatnonzero(
            (self.origins_all >= first_valid_origin)
            & (self.origins_all <= last_valid_origin)
        )
        if selected.size == 0:
            raise ValueError(f"No complete samples found for year {self.year}")

        # Keep only the requested year in memory and use direct integer offsets.
        year_wind = wind_frame.loc[
            (wind_frame.index >= year_start) & (wind_frame.index < year_end),
            self.provinces,
        ]
        expected_index = pd.date_range(
            year_start, year_end - pd.Timedelta(hours=1), freq="h"
        )
        if not year_wind.index.equals(expected_index):
            missing_hours = expected_index.difference(year_wind.index)
            raise ValueError(
                f"Wind data is not a complete hourly series for {self.year}; "
                f"missing {len(missing_hours)} hour(s)"
            )

        self.wind = year_wind.to_numpy(dtype=np.float32, copy=True)
        self.origins = self.origins_all[selected]
        self.forecast = np.asarray(forecast[selected], dtype=np.float32)
        self.origin_hour_offsets = (
            (self.origins - year_start) / pd.Timedelta(hours=1)
        ).astype(np.int64).to_numpy()

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> dict[str, Any]:
        origin_offset = int(self.origin_hour_offsets[index])
        history_start = origin_offset - self.history_hours + 1
        target_start = origin_offset + 1
        target_end = target_start + self.forecast_hours

        return {
            "history_wind": torch.from_numpy(
                self.wind[history_start : origin_offset + 1].copy()
            ),
            "future_weather": torch.from_numpy(self.forecast[index].copy()),
            "target_wind": torch.from_numpy(
                self.wind[target_start:target_end].copy()
            ),
            "origin_ns": torch.tensor(
                self.origins[index].value, dtype=torch.int64
            ),
        }


def create_wind_dataloader(
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    year: int = 2019,
    **dataset_kwargs: Any,
) -> DataLoader:
    """Create the training DataLoader; 2019 is the default and intended year."""

    dataset = WindForecastDataset(year=year, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    loader = create_wind_dataloader(batch_size=16, shuffle=False)
    batch = next(iter(loader))
    print(f"samples: {len(loader.dataset)}")
    print(f"history_wind: {tuple(batch['history_wind'].shape)}")
    print(f"future_weather: {tuple(batch['future_weather'].shape)}")
    print(f"target_wind: {tuple(batch['target_wind'].shape)}")

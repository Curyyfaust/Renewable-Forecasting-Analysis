"""Neural baselines for 24-hour provincial wind-power forecasting.

All models share one interface::

    prediction = model(history_wind, future_weather)

Inputs:
    history_wind:   [batch, 72, provinces]
    future_weather: [batch, 24, provinces, weather_features]

Output:
    prediction:     [batch, 24, provinces]

Each province is encoded with shared model weights. This avoids tying the
architecture to exactly 31 provinces while still producing one forecast per
province.
"""

from __future__ import annotations

import math

import torch
from torch import nn


HISTORY_HOURS = 72
FORECAST_HOURS = 24
WEATHER_FEATURES = 6


class WindForecastModel(nn.Module):
    """Base class providing common input validation and reshaping."""

    def __init__(
        self,
        history_hours: int = HISTORY_HOURS,
        forecast_hours: int = FORECAST_HOURS,
        weather_features: int = WEATHER_FEATURES,
    ) -> None:
        super().__init__()
        self.history_hours = history_hours
        self.forecast_hours = forecast_hours
        self.weather_features = weather_features

    def _validate_inputs(
        self, history_wind: torch.Tensor, future_weather: torch.Tensor
    ) -> tuple[int, int]:
        if history_wind.ndim != 3:
            raise ValueError(
                "history_wind must have shape [batch, history_hours, provinces]"
            )
        if future_weather.ndim != 4:
            raise ValueError(
                "future_weather must have shape "
                "[batch, forecast_hours, provinces, weather_features]"
            )
        batch, history_hours, provinces = history_wind.shape
        weather_batch, forecast_hours, weather_provinces, features = (
            future_weather.shape
        )
        if history_hours != self.history_hours:
            raise ValueError(
                f"Expected {self.history_hours} history hours, got {history_hours}"
            )
        if forecast_hours != self.forecast_hours:
            raise ValueError(
                f"Expected {self.forecast_hours} forecast hours, got {forecast_hours}"
            )
        if features != self.weather_features:
            raise ValueError(
                f"Expected {self.weather_features} weather features, got {features}"
            )
        if batch != weather_batch or provinces != weather_provinces:
            raise ValueError("History and weather batch/province dimensions must match")
        return batch, provinces

    @staticmethod
    def _province_first(
        history_wind: torch.Tensor, future_weather: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert inputs to independent province sequences in the batch axis."""
        batch, _, provinces = history_wind.shape
        history = history_wind.permute(0, 2, 1).reshape(batch * provinces, -1, 1)
        weather = future_weather.permute(0, 2, 1, 3).reshape(
            batch * provinces, future_weather.shape[1], future_weather.shape[3]
        )
        return history, weather

    def _restore_output(
        self, prediction: torch.Tensor, batch: int, provinces: int
    ) -> torch.Tensor:
        return prediction.reshape(batch, provinces, self.forecast_hours).permute(0, 2, 1)


class DNNModel(WindForecastModel):
    """Multilayer perceptron over flattened history and future weather."""

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        **kwargs: int,
    ) -> None:
        super().__init__(**kwargs)
        input_dim = self.history_hours + self.forecast_hours * self.weather_features
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.forecast_hours),
        )

    def forward(
        self, history_wind: torch.Tensor, future_weather: torch.Tensor
    ) -> torch.Tensor:
        batch, provinces = self._validate_inputs(history_wind, future_weather)
        history, weather = self._province_first(history_wind, future_weather)
        features = torch.cat((history.flatten(1), weather.flatten(1)), dim=1)
        return self._restore_output(self.network(features), batch, provinces)


class CNNModel(WindForecastModel):
    """Temporal CNN with separate history and weather encoders."""

    def __init__(
        self,
        channels: int = 64,
        dropout: float = 0.1,
        **kwargs: int,
    ) -> None:
        super().__init__(**kwargs)
        self.history_encoder = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.weather_encoder = nn.Sequential(
            nn.Conv1d(self.weather_features, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(channels * 2, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, self.forecast_hours),
        )

    def forward(
        self, history_wind: torch.Tensor, future_weather: torch.Tensor
    ) -> torch.Tensor:
        batch, provinces = self._validate_inputs(history_wind, future_weather)
        history, weather = self._province_first(history_wind, future_weather)
        history_code = self.history_encoder(history.transpose(1, 2)).squeeze(-1)
        weather_code = self.weather_encoder(weather.transpose(1, 2)).squeeze(-1)
        prediction = self.decoder(torch.cat((history_code, weather_code), dim=1))
        return self._restore_output(prediction, batch, provinces)


class LSTMModel(WindForecastModel):
    """Two-stream LSTM encoder for wind history and future weather."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs: int,
    ) -> None:
        super().__init__(**kwargs)
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.history_lstm = nn.LSTM(
            1,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.weather_lstm = nn.LSTM(
            self.weather_features,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.forecast_hours),
        )

    def forward(
        self, history_wind: torch.Tensor, future_weather: torch.Tensor
    ) -> torch.Tensor:
        batch, provinces = self._validate_inputs(history_wind, future_weather)
        history, weather = self._province_first(history_wind, future_weather)
        _, (history_hidden, _) = self.history_lstm(history)
        _, (weather_hidden, _) = self.weather_lstm(weather)
        code = torch.cat((history_hidden[-1], weather_hidden[-1]), dim=1)
        return self._restore_output(self.decoder(code), batch, provinces)


class SinusoidalPositionEncoding(nn.Module):
    """Fixed positional encoding with no dependence on training data length."""

    def __init__(self, model_dim: int, max_length: int) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / model_dim)
        )
        encoding = torch.zeros(max_length, model_dim)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[:, : values.shape[1]]


class TransformerModel(WindForecastModel):
    """Transformer over 72 history tokens followed by 24 forecast tokens."""

    def __init__(
        self,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
        **kwargs: int,
    ) -> None:
        super().__init__(**kwargs)
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.history_projection = nn.Linear(1, model_dim)
        self.weather_projection = nn.Linear(self.weather_features, model_dim)
        self.segment_embedding = nn.Parameter(torch.zeros(2, model_dim))
        self.position = SinusoidalPositionEncoding(
            model_dim, self.history_hours + self.forecast_hours
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_projection = nn.Linear(model_dim, 1)

    def forward(
        self, history_wind: torch.Tensor, future_weather: torch.Tensor
    ) -> torch.Tensor:
        batch, provinces = self._validate_inputs(history_wind, future_weather)
        history, weather = self._province_first(history_wind, future_weather)
        history_tokens = self.history_projection(history) + self.segment_embedding[0]
        weather_tokens = self.weather_projection(weather) + self.segment_embedding[1]
        tokens = self.position(torch.cat((history_tokens, weather_tokens), dim=1))
        encoded = self.encoder(tokens)
        forecast_tokens = encoded[:, -self.forecast_hours :]
        prediction = self.output_projection(forecast_tokens).squeeze(-1)
        return self._restore_output(prediction, batch, provinces)


MODEL_REGISTRY: dict[str, type[WindForecastModel]] = {
    "cnn": CNNModel,
    "lstm": LSTMModel,
    "transformer": TransformerModel,
    "dnn": DNNModel,
}


def build_model(name: str, **kwargs: int | float) -> WindForecastModel:
    """Build one of the four models by a case-insensitive name."""
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](**kwargs)


if __name__ == "__main__":
    example_history = torch.randn(2, HISTORY_HOURS, 31)
    example_weather = torch.randn(2, FORECAST_HOURS, 31, WEATHER_FEATURES)
    for model_name in MODEL_REGISTRY:
        model = build_model(model_name)
        result = model(example_history, example_weather)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"{model_name:11s} output={tuple(result.shape)} params={parameter_count:,}")

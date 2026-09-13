from __future__ import annotations

import math

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1]).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        window_size: int = 30,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(input_size, d_model)
        self.register_buffer(
            "position_encoding",
            self._sinusoidal_encoding(window_size, d_model),
            persistent=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    @staticmethod
    def _sinusoidal_encoding(window_size: int, d_model: int) -> torch.Tensor:
        position = torch.arange(window_size, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(1, window_size, d_model)
        encoding[0, :, 0::2] = torch.sin(position * divisor)
        encoding[0, :, 1::2] = torch.cos(position * divisor)
        return encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.projection(x) + self.position_encoding[:, : x.shape[1]]
        encoded = self.encoder(encoded)
        return self.head(self.norm(encoded[:, -1])).squeeze(-1)


def create_model(name: str, input_size: int, window_size: int, config: dict) -> nn.Module:
    if name == "lstm":
        return LSTMRegressor(
            input_size=input_size,
            hidden_size=int(config.get("hidden_size", 64)),
            num_layers=int(config.get("num_layers", 2)),
            dropout=float(config.get("dropout", 0.2)),
        )
    if name == "transformer":
        return TransformerRegressor(
            input_size=input_size,
            window_size=window_size,
            d_model=int(config.get("d_model", 64)),
            num_heads=int(config.get("num_heads", 4)),
            num_layers=int(config.get("num_layers", 2)),
            dim_feedforward=int(config.get("dim_feedforward", 128)),
            dropout=float(config.get("dropout", 0.1)),
        )
    raise ValueError(f"Unknown model: {name}")

"""Configuration loader for sports prediction system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class NFLConfig:
    seasons: list[int] = field(default_factory=lambda: [2021, 2022, 2023, 2024, 2025])
    elo_k_factor: float = 20.0
    elo_home_advantage: float = 48.0
    elo_mean: float = 1500.0
    epa_smoothing_window: int = 4


@dataclass(frozen=True)
class NBAConfig:
    seasons: list[str] = field(default_factory=lambda: ["2023-24", "2024-25", "2025-26"])
    net_rating_window: int = 10
    pace_league_avg: float = 100.0
    rest_penalty_b2b: float = -2.5
    travel_penalty_per_1000mi: float = -0.5


@dataclass(frozen=True)
class MLBConfig:
    seasons: list[int] = field(default_factory=lambda: [2023, 2024, 2025])
    xwoba_smoothing_pa: int = 150
    park_factor_regress_games: int = 81


@dataclass(frozen=True)
class ModelConfig:
    monte_carlo_iterations: int = 10000
    calibration_bins: int = 20
    bayesian_chains: int = 4
    bayesian_draws: int = 2000


@dataclass(frozen=True)
class EdgeConfig:
    min_edge_pct: float = 3.0
    kelly_fraction: float = 0.25
    max_position_pct: float = 5.0
    sharp_move_threshold: float = 2.0


@dataclass(frozen=True)
class MarketConfig:
    kalshi_base_url: str = "https://trading-api.kalshi.com/trade-api/v2"
    polymarket_base_url: str = "https://clob.polymarket.com"
    price_refresh_seconds: int = 60


@dataclass(frozen=True)
class DataConfig:
    cache_dir: str = ".cache/sports_data"
    cache_ttl_hours: int = 6


@dataclass(frozen=True)
class Settings:
    data: DataConfig = field(default_factory=DataConfig)
    nfl: NFLConfig = field(default_factory=NFLConfig)
    nba: NBAConfig = field(default_factory=NBAConfig)
    mlb: MLBConfig = field(default_factory=MLBConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    market: MarketConfig = field(default_factory=MarketConfig)


def _build_dataclass(cls, data: dict[str, Any] | None):
    if data is None:
        return cls()
    filtered = {k: v for k, v in data.items() if k in {f.name for f in cls.__dataclass_fields__.values()}}
    return cls(**filtered)


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from YAML config file, with environment variable overrides."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"

    if config_path.exists():
        raw = _load_yaml(config_path)
    else:
        raw = {}

    # Environment variable overrides for sensitive values
    env_overrides = {
        "KALSHI_API_KEY": ("market", "kalshi_api_key"),
        "POLYMARKET_API_KEY": ("market", "polymarket_api_key"),
        "NOAA_API_TOKEN": ("weather", "noaa_token"),
    }
    for env_var, (section, key) in env_overrides.items():
        val = os.environ.get(env_var)
        if val:
            raw.setdefault(section, {})[key] = val

    return Settings(
        data=_build_dataclass(DataConfig, raw.get("data")),
        nfl=_build_dataclass(NFLConfig, raw.get("nfl")),
        nba=_build_dataclass(NBAConfig, raw.get("nba")),
        mlb=_build_dataclass(MLBConfig, raw.get("mlb")),
        models=_build_dataclass(ModelConfig, raw.get("models")),
        edge=_build_dataclass(EdgeConfig, raw.get("edge_detection")),
        market=_build_dataclass(MarketConfig, raw.get("market")),
    )

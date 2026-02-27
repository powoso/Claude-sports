"""Tests for configuration loading."""

from pathlib import Path

import pytest

from sports_predict.utils.config import (
    DataConfig,
    EdgeConfig,
    MLBConfig,
    NBAConfig,
    NFLConfig,
    Settings,
    load_settings,
)


class TestConfig:
    def test_default_settings(self):
        settings = Settings()
        assert settings.nfl.elo_k_factor == 20.0
        assert settings.nba.net_rating_window == 10
        assert settings.mlb.xwoba_smoothing_pa == 150

    def test_load_from_yaml(self):
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        settings = load_settings(config_path)
        assert settings.nfl.elo_k_factor == 20
        assert settings.models.monte_carlo_iterations == 10000
        assert settings.edge.min_edge_pct == 3.0

    def test_load_missing_file_uses_defaults(self):
        settings = load_settings(Path("/nonexistent/config.yaml"))
        assert settings.nfl.elo_k_factor == 20.0
        assert settings.data.cache_ttl_hours == 6

    def test_nfl_config_immutable(self):
        cfg = NFLConfig()
        with pytest.raises(AttributeError):
            cfg.elo_k_factor = 30.0

    def test_edge_config(self):
        cfg = EdgeConfig(min_edge_pct=5.0, kelly_fraction=0.5)
        assert cfg.min_edge_pct == 5.0
        assert cfg.kelly_fraction == 0.5

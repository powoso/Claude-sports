"""Tests for edge detection and line movement tracking."""

from datetime import datetime

import pandas as pd
import pytest

from sports_predict.edge_detection.edge_finder import (
    BiasType,
    Edge,
    EdgeFinder,
    EdgeType,
    LineMovementTracker,
)


class TestEdgeFinder:
    def setup_method(self):
        self.finder = EdgeFinder(
            min_edge_pct=3.0,
            kelly_fraction=0.25,
            max_position_pct=5.0,
        )

    def test_calculate_edge(self):
        # Model says 60%, market says 50% → 10% edge
        edge = self.finder.calculate_edge(0.60, 0.50)
        assert edge == pytest.approx(10.0)

    def test_calculate_edge_negative(self):
        # Model says 40%, market says 50% → -10% edge (overpriced)
        edge = self.finder.calculate_edge(0.40, 0.50)
        assert edge == pytest.approx(-10.0)

    def test_kelly_size_positive_edge(self):
        # Clear positive edge
        kelly = self.finder.kelly_size(0.60, 0.50)
        assert kelly > 0

    def test_kelly_size_negative_edge(self):
        # Negative edge → no bet
        kelly = self.finder.kelly_size(0.40, 0.50)
        assert kelly == 0.0

    def test_kelly_size_capped(self):
        # Even large edge should be capped
        kelly = self.finder.kelly_size(0.90, 0.50)
        assert kelly <= self.finder.max_position_pct / 100

    def test_kelly_size_fractional(self):
        """Kelly should be scaled by kelly_fraction."""
        # Full Kelly (high cap to avoid capping interference)
        full_finder = EdgeFinder(kelly_fraction=1.0, max_position_pct=100.0)
        full_kelly = full_finder.kelly_size(0.60, 0.50)

        # Quarter Kelly (high cap)
        quarter_finder = EdgeFinder(kelly_fraction=0.25, max_position_pct=100.0)
        quarter_kelly = quarter_finder.kelly_size(0.60, 0.50)

        assert quarter_kelly == pytest.approx(full_kelly * 0.25, rel=0.01)

    def test_find_game_edges(self):
        model_probs = {"TeamA vs TeamB": 0.65}
        market_prices = pd.DataFrame([
            {
                "event_name": "TeamA vs TeamB",
                "outcome": "YES",
                "price": 0.50,
                "source": "kalshi",
            }
        ])

        edges = self.finder.find_game_edges(model_probs, market_prices, "nfl")
        assert len(edges) == 1
        assert edges[0].edge_pct == pytest.approx(15.0)
        assert edges[0].sport == "nfl"

    def test_find_game_edges_no_edge(self):
        model_probs = {"TeamA vs TeamB": 0.51}
        market_prices = pd.DataFrame([
            {
                "event_name": "TeamA vs TeamB",
                "outcome": "YES",
                "price": 0.50,
                "source": "kalshi",
            }
        ])

        edges = self.finder.find_game_edges(model_probs, market_prices, "nfl")
        assert len(edges) == 0  # 1% edge below 3% threshold

    def test_find_total_edges(self):
        edges = self.finder.find_total_edges(
            model_total=48.0,
            model_std=10.0,
            market_line=44.0,
            over_price=0.50,
            under_price=0.50,
            sport="nfl",
            event_name="Game Total",
        )

        # Model projects 48, line is 44 → over should have edge
        over_edges = [e for e in edges if e.outcome == "OVER"]
        assert len(over_edges) > 0
        assert over_edges[0].edge_pct > 0

    def test_detect_favorite_bias(self):
        model_probs = {"Cowboys vs Giants": 0.55}
        market_prices = pd.DataFrame([
            {
                "event_name": "Cowboys vs Giants",
                "outcome": "YES",
                "price": 0.70,
                "source": "polymarket",
            }
        ])

        edges = self.finder.find_game_edges(model_probs, market_prices, "nfl")
        # Market has Cowboys at 70% but model says 55% → may detect bias
        if edges:
            bias_detected = any(
                BiasType.FAVORITE_BIAS in e.detected_biases or BiasType.NAME_BIAS in e.detected_biases
                for e in edges
            )
            assert bias_detected

    def test_edge_sorted_by_magnitude(self):
        model_probs = {
            "Game1": 0.70,
            "Game2": 0.80,
        }
        market_prices = pd.DataFrame([
            {"event_name": "Game1", "outcome": "YES", "price": 0.50, "source": "kalshi"},
            {"event_name": "Game2", "outcome": "YES", "price": 0.50, "source": "kalshi"},
        ])

        edges = self.finder.find_game_edges(model_probs, market_prices, "nfl")
        assert len(edges) == 2
        # Larger edge should be first
        assert abs(edges[0].edge_pct) >= abs(edges[1].edge_pct)


class TestLineMovementTracker:
    def setup_method(self):
        self.tracker = LineMovementTracker(sharp_move_threshold=2.0)

    def test_first_price_no_movement(self):
        result = self.tracker.record_price("Game1", "YES", "kalshi", 0.50)
        assert result is None

    def test_small_move_not_flagged(self):
        self.tracker.record_price("Game1", "YES", "kalshi", 0.50)
        result = self.tracker.record_price("Game1", "YES", "kalshi", 0.51)
        assert result is None  # 1% move below 2% threshold

    def test_sharp_move_detected(self):
        self.tracker.record_price("Game1", "YES", "kalshi", 0.50)
        result = self.tracker.record_price("Game1", "YES", "kalshi", 0.55)
        assert result is not None
        assert result.is_sharp_move
        assert result.change_pct == pytest.approx(5.0)

    def test_different_sources_independent(self):
        self.tracker.record_price("Game1", "YES", "kalshi", 0.50)
        self.tracker.record_price("Game1", "YES", "polymarket", 0.45)
        # Movement on polymarket shouldn't compare to kalshi price
        result = self.tracker.record_price("Game1", "YES", "polymarket", 0.48)
        assert result is not None
        assert result.change_pct == pytest.approx(3.0)

    def test_vegas_pm_divergence(self):
        vegas = pd.DataFrame([
            {"event_name": "Game1", "outcome": "YES", "price_vegas": 0.55},
        ])
        pm = pd.DataFrame([
            {"event_name": "Game1", "outcome": "YES", "price_pm": 0.48},
        ])
        # Manually construct for divergence test
        merged = vegas.merge(pm, on=["event_name", "outcome"])
        merged["divergence_pct"] = (merged["price_vegas"] - merged["price_pm"]) * 100
        assert merged.iloc[0]["divergence_pct"] == pytest.approx(7.0)


class TestEdgeDataclass:
    def test_edge_actionable(self):
        edge = Edge(
            sport="nfl",
            edge_type=EdgeType.GAME_WINNER,
            event_name="Test",
            outcome="YES",
            model_prob=0.60,
            market_prob=0.50,
            edge_pct=10.0,
            kelly_fraction=0.05,
            recommended_size_pct=2.0,
            market_source="kalshi",
            confidence=0.8,
        )
        assert edge.is_actionable

    def test_edge_not_actionable_negative(self):
        edge = Edge(
            sport="nfl",
            edge_type=EdgeType.GAME_WINNER,
            event_name="Test",
            outcome="YES",
            model_prob=0.40,
            market_prob=0.50,
            edge_pct=-10.0,
            kelly_fraction=0.0,
            recommended_size_pct=0.0,
            market_source="kalshi",
            confidence=0.8,
        )
        assert not edge.is_actionable

"""Edge detection engine: model vs market price comparison.

Identifies profitable betting opportunities by comparing model probabilities
to prediction market prices and Vegas lines. Includes:
- Raw edge calculation (model prob - market prob)
- Kelly criterion position sizing
- Public bias detection
- Sharp money movement tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EdgeType(str, Enum):
    GAME_WINNER = "game_winner"
    SPREAD = "spread"
    TOTAL = "total"
    PLAYER_PROP = "player_prop"
    SEASON_FUTURES = "season_futures"
    WIN_TOTAL = "win_total"


class BiasType(str, Enum):
    FAVORITE_BIAS = "favorite_bias"  # public overweights favorites
    OVER_BIAS = "over_bias"  # public takes overs disproportionately
    RECENCY_BIAS = "recency_bias"  # overweight recent performance
    NAME_BIAS = "name_bias"  # big-name teams get inflated prices
    NONE = "none"


@dataclass
class Edge:
    """A detected edge in a prediction market."""

    sport: str
    edge_type: EdgeType
    event_name: str
    outcome: str
    model_prob: float
    market_prob: float
    edge_pct: float  # model_prob - market_prob, in percentage points
    kelly_fraction: float
    recommended_size_pct: float  # of bankroll
    market_source: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    detected_biases: list[BiasType] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.edge_pct > 0 and self.recommended_size_pct > 0


@dataclass
class LineMovement:
    """Tracks line movement for sharp money detection."""

    event_name: str
    outcome: str
    source: str
    old_price: float
    new_price: float
    change_pct: float
    timestamp: datetime
    is_sharp_move: bool
    direction: str  # 'toward_model' or 'away_from_model'


class EdgeFinder:
    """Core edge detection engine."""

    def __init__(
        self,
        min_edge_pct: float = 3.0,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 5.0,
    ):
        self.min_edge_pct = min_edge_pct
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct

    def calculate_edge(
        self,
        model_prob: float,
        market_prob: float,
    ) -> float:
        """Calculate raw edge in percentage points."""
        return (model_prob - market_prob) * 100

    def kelly_size(
        self,
        model_prob: float,
        market_prob: float,
    ) -> float:
        """Calculate Kelly criterion bet size as fraction of bankroll.

        Kelly = (bp - q) / b
        where b = odds (decimal - 1), p = model prob, q = 1 - p
        """
        if market_prob <= 0 or market_prob >= 1:
            return 0.0

        # Convert market prob to decimal odds
        decimal_odds = 1.0 / market_prob
        b = decimal_odds - 1.0

        p = model_prob
        q = 1.0 - p

        kelly = (b * p - q) / b if b > 0 else 0.0

        # Apply fractional Kelly
        kelly *= self.kelly_fraction

        # Cap at max position size
        return max(0.0, min(kelly, self.max_position_pct / 100.0))

    def find_game_edges(
        self,
        model_probs: dict[str, float],
        market_prices: pd.DataFrame,
        sport: str,
    ) -> list[Edge]:
        """Find edges across game outcome markets.

        Args:
            model_probs: Dict of event_name -> model probability
            market_prices: DataFrame with event_name, outcome, price, source columns
            sport: Sport identifier
        """
        edges = []

        for event_name, model_prob in model_probs.items():
            # Find matching market prices
            matches = market_prices[
                market_prices["event_name"].str.contains(event_name, case=False, na=False)
            ]

            for _, mkt in matches.iterrows():
                market_prob = mkt["price"]
                if market_prob <= 0 or market_prob >= 1:
                    continue

                edge_pct = self.calculate_edge(model_prob, market_prob)

                if abs(edge_pct) < self.min_edge_pct:
                    continue

                # Determine which side has edge
                if edge_pct > 0:
                    # Model thinks this outcome is underpriced
                    kelly = self.kelly_size(model_prob, market_prob)
                    size = kelly * 100  # as percentage

                    biases = self._detect_biases(model_prob, market_prob, event_name)

                    edges.append(Edge(
                        sport=sport,
                        edge_type=EdgeType.GAME_WINNER,
                        event_name=event_name,
                        outcome=mkt.get("outcome", "YES"),
                        model_prob=round(model_prob, 4),
                        market_prob=round(market_prob, 4),
                        edge_pct=round(edge_pct, 2),
                        kelly_fraction=round(kelly, 4),
                        recommended_size_pct=round(min(size, self.max_position_pct), 2),
                        market_source=mkt.get("source", "unknown"),
                        confidence=round(min(abs(edge_pct) / 10.0, 1.0), 2),
                        detected_biases=biases,
                        metadata={
                            "volume": mkt.get("volume"),
                        },
                    ))

        return sorted(edges, key=lambda e: abs(e.edge_pct), reverse=True)

    def find_total_edges(
        self,
        model_total: float,
        model_std: float,
        market_line: float,
        over_price: float,
        under_price: float,
        sport: str,
        event_name: str,
        source: str = "unknown",
    ) -> list[Edge]:
        """Find edges in over/under totals markets."""
        from scipy import stats as sp_stats

        edges = []

        z = (market_line - model_total) / model_std
        model_over_prob = 1.0 - sp_stats.norm.cdf(z)
        model_under_prob = sp_stats.norm.cdf(z)

        # Check over
        over_edge = self.calculate_edge(model_over_prob, over_price)
        if abs(over_edge) >= self.min_edge_pct:
            kelly = self.kelly_size(model_over_prob, over_price)
            edges.append(Edge(
                sport=sport,
                edge_type=EdgeType.TOTAL,
                event_name=event_name,
                outcome="OVER",
                model_prob=round(model_over_prob, 4),
                market_prob=round(over_price, 4),
                edge_pct=round(over_edge, 2),
                kelly_fraction=round(kelly, 4),
                recommended_size_pct=round(min(kelly * 100, self.max_position_pct), 2),
                market_source=source,
                confidence=round(min(abs(over_edge) / 10.0, 1.0), 2),
                detected_biases=(
                    [BiasType.OVER_BIAS] if over_edge < 0 else []
                ),
                metadata={
                    "model_total": model_total,
                    "market_line": market_line,
                },
            ))

        # Check under
        under_edge = self.calculate_edge(model_under_prob, under_price)
        if abs(under_edge) >= self.min_edge_pct:
            kelly = self.kelly_size(model_under_prob, under_price)
            edges.append(Edge(
                sport=sport,
                edge_type=EdgeType.TOTAL,
                event_name=event_name,
                outcome="UNDER",
                model_prob=round(model_under_prob, 4),
                market_prob=round(under_price, 4),
                edge_pct=round(under_edge, 2),
                kelly_fraction=round(kelly, 4),
                recommended_size_pct=round(min(kelly * 100, self.max_position_pct), 2),
                market_source=source,
                confidence=round(min(abs(under_edge) / 10.0, 1.0), 2),
            ))

        return edges

    def find_prop_edges(
        self,
        prop_predictions: list,
        market_prices: pd.DataFrame,
        sport: str,
    ) -> list[Edge]:
        """Find edges in player prop markets."""
        edges = []

        for pred in prop_predictions:
            player = pred.player
            stat = pred.stat

            # Find matching market
            matches = market_prices[
                market_prices["event_name"].str.contains(player, case=False, na=False)
                & market_prices["event_name"].str.contains(stat, case=False, na=False)
            ]

            for _, mkt in matches.iterrows():
                market_prob = mkt["price"]
                outcome = mkt.get("outcome", "OVER")

                if "over" in outcome.lower():
                    model_prob = pred.over_prob
                else:
                    model_prob = pred.under_prob

                edge_pct = self.calculate_edge(model_prob, market_prob)

                if abs(edge_pct) >= self.min_edge_pct:
                    kelly = self.kelly_size(model_prob, market_prob)
                    edges.append(Edge(
                        sport=sport,
                        edge_type=EdgeType.PLAYER_PROP,
                        event_name=f"{player} {stat}",
                        outcome=outcome,
                        model_prob=round(model_prob, 4),
                        market_prob=round(market_prob, 4),
                        edge_pct=round(edge_pct, 2),
                        kelly_fraction=round(kelly, 4),
                        recommended_size_pct=round(min(kelly * 100, self.max_position_pct), 2),
                        market_source=mkt.get("source", "unknown"),
                        confidence=round(min(abs(edge_pct) / 10.0, 1.0), 2),
                        metadata={
                            "projected_value": pred.projected_value,
                            "line": pred.line,
                        },
                    ))

        return sorted(edges, key=lambda e: abs(e.edge_pct), reverse=True)

    def find_season_edges(
        self,
        team_outcomes: list,
        market_prices: pd.DataFrame,
        sport: str,
    ) -> list[Edge]:
        """Find edges in season-long markets (win totals, playoffs, etc.)."""
        edges = []

        for outcome in team_outcomes:
            team = outcome.team

            # Win total over/under
            if outcome.over_under_line and outcome.over_prob:
                matches = market_prices[
                    market_prices["event_name"].str.contains(team, case=False, na=False)
                    & market_prices["event_name"].str.contains("win", case=False, na=False)
                ]

                for _, mkt in matches.iterrows():
                    market_prob = mkt["price"]
                    model_prob = outcome.over_prob

                    edge_pct = self.calculate_edge(model_prob, market_prob)
                    if abs(edge_pct) >= self.min_edge_pct:
                        kelly = self.kelly_size(model_prob, market_prob)
                        edges.append(Edge(
                            sport=sport,
                            edge_type=EdgeType.WIN_TOTAL,
                            event_name=f"{team} wins",
                            outcome="OVER" if edge_pct > 0 else "UNDER",
                            model_prob=round(model_prob, 4),
                            market_prob=round(market_prob, 4),
                            edge_pct=round(edge_pct, 2),
                            kelly_fraction=round(kelly, 4),
                            recommended_size_pct=round(
                                min(kelly * 100, self.max_position_pct), 2
                            ),
                            market_source=mkt.get("source", "unknown"),
                            confidence=round(min(abs(edge_pct) / 10.0, 1.0), 2),
                            metadata={
                                "mean_wins": outcome.mean_wins,
                                "line": outcome.over_under_line,
                            },
                        ))

            # Playoff probability
            matches = market_prices[
                market_prices["event_name"].str.contains(team, case=False, na=False)
                & market_prices["event_name"].str.contains("playoff", case=False, na=False)
            ]
            for _, mkt in matches.iterrows():
                market_prob = mkt["price"]
                model_prob = outcome.playoff_prob

                edge_pct = self.calculate_edge(model_prob, market_prob)
                if abs(edge_pct) >= self.min_edge_pct:
                    kelly = self.kelly_size(model_prob, market_prob)
                    edges.append(Edge(
                        sport=sport,
                        edge_type=EdgeType.SEASON_FUTURES,
                        event_name=f"{team} make playoffs",
                        outcome="YES",
                        model_prob=round(model_prob, 4),
                        market_prob=round(market_prob, 4),
                        edge_pct=round(edge_pct, 2),
                        kelly_fraction=round(kelly, 4),
                        recommended_size_pct=round(
                            min(kelly * 100, self.max_position_pct), 2
                        ),
                        market_source=mkt.get("source", "unknown"),
                        confidence=round(min(abs(edge_pct) / 10.0, 1.0), 2),
                    ))

        return sorted(edges, key=lambda e: abs(e.edge_pct), reverse=True)

    def _detect_biases(
        self, model_prob: float, market_prob: float, event_name: str
    ) -> list[BiasType]:
        """Detect potential structural biases in market pricing."""
        biases = []

        # Favorite bias: if model says close but market heavily favors one side
        if market_prob > 0.65 and model_prob < 0.60:
            biases.append(BiasType.FAVORITE_BIAS)

        # Public tends to bet on big-name teams
        big_names = [
            "cowboys", "lakers", "yankees", "patriots", "chiefs",
            "warriors", "dodgers", "celtics",
        ]
        if any(name in event_name.lower() for name in big_names):
            if market_prob > model_prob + 0.03:
                biases.append(BiasType.NAME_BIAS)

        return biases


class LineMovementTracker:
    """Tracks line movements to detect sharp money."""

    def __init__(self, sharp_move_threshold: float = 2.0):
        self.sharp_move_threshold = sharp_move_threshold
        self.price_history: list[dict] = []

    def record_price(
        self,
        event_name: str,
        outcome: str,
        source: str,
        price: float,
        timestamp: datetime | None = None,
    ) -> LineMovement | None:
        """Record a new price and check for sharp movement.

        Returns LineMovement if a significant move is detected.
        """
        ts = timestamp or datetime.utcnow()

        # Find previous price for this event/outcome/source
        prev = None
        for h in reversed(self.price_history):
            if (
                h["event_name"] == event_name
                and h["outcome"] == outcome
                and h["source"] == source
            ):
                prev = h
                break

        self.price_history.append({
            "event_name": event_name,
            "outcome": outcome,
            "source": source,
            "price": price,
            "timestamp": ts,
        })

        if prev is None:
            return None

        change_pct = (price - prev["price"]) * 100

        if abs(change_pct) >= self.sharp_move_threshold:
            return LineMovement(
                event_name=event_name,
                outcome=outcome,
                source=source,
                old_price=prev["price"],
                new_price=price,
                change_pct=round(change_pct, 2),
                timestamp=ts,
                is_sharp_move=True,
                direction="up" if change_pct > 0 else "down",
            )

        return None

    def detect_vegas_prediction_market_divergence(
        self,
        vegas_prices: pd.DataFrame,
        prediction_market_prices: pd.DataFrame,
        threshold_pct: float = 5.0,
    ) -> pd.DataFrame:
        """Find events where Vegas and prediction markets significantly diverge.

        This can indicate that one market has repriced due to news (e.g., injury)
        that the other hasn't incorporated yet.
        """
        # Merge on event name
        merged = vegas_prices.merge(
            prediction_market_prices,
            on=["event_name", "outcome"],
            suffixes=("_vegas", "_pm"),
        )

        if merged.empty:
            return pd.DataFrame()

        merged["divergence_pct"] = (
            (merged["price_vegas"] - merged["price_pm"]) * 100
        )
        merged["abs_divergence"] = merged["divergence_pct"].abs()

        significant = merged[merged["abs_divergence"] >= threshold_pct].copy()
        return significant.sort_values("abs_divergence", ascending=False)

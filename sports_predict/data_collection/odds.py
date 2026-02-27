"""Odds and prediction market data collection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

import httpx
import pandas as pd

if TYPE_CHECKING:
    from sports_predict.utils.cache import DataCache

logger = logging.getLogger(__name__)


class Sport(str, Enum):
    NFL = "nfl"
    NBA = "nba"
    MLB = "mlb"


class MarketType(str, Enum):
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"
    PROP = "prop"
    FUTURES = "futures"


@dataclass
class MarketPrice:
    """Represents a single market price snapshot."""

    sport: Sport
    market_type: MarketType
    event_name: str
    outcome: str
    price: float  # probability 0-1
    source: str  # 'kalshi', 'polymarket', 'vegas'
    timestamp: datetime
    raw_price: float | None = None  # original odds format
    volume: float | None = None
    metadata: dict | None = None


class OddsCollector:
    """Collects odds from Vegas, Kalshi, and Polymarket."""

    def __init__(self, cache: DataCache | None = None):
        self.cache = cache
        self._client = httpx.Client(timeout=30.0)

    def american_to_prob(self, odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds > 0:
            return 100.0 / (odds + 100.0)
        else:
            return abs(odds) / (abs(odds) + 100.0)

    def decimal_to_prob(self, odds: float) -> float:
        """Convert decimal odds to implied probability."""
        return 1.0 / odds if odds > 0 else 0.0

    def remove_vig(self, prob_a: float, prob_b: float) -> tuple[float, float]:
        """Remove vigorish from a two-way market to get fair probabilities."""
        total = prob_a + prob_b
        if total == 0:
            return 0.5, 0.5
        return prob_a / total, prob_b / total

    def fetch_kalshi_markets(
        self, sport: Sport, status: str = "open"
    ) -> list[MarketPrice]:
        """Fetch active sports markets from Kalshi.

        Note: Requires KALSHI_API_KEY environment variable for authenticated access.
        """
        # Kalshi API v2 market listing
        search_terms = {
            Sport.NFL: "NFL",
            Sport.NBA: "NBA",
            Sport.MLB: "MLB",
        }

        try:
            resp = self._client.get(
                "https://trading-api.kalshi.com/trade-api/v2/markets",
                params={
                    "status": status,
                    "series_ticker": search_terms.get(sport, ""),
                    "limit": 100,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            logger.warning("Kalshi API error: %s", e)
            return []

        markets = []
        now = datetime.utcnow()
        for mkt in data.get("markets", []):
            yes_price = mkt.get("yes_ask", mkt.get("last_price", 0)) / 100.0
            markets.append(
                MarketPrice(
                    sport=sport,
                    market_type=MarketType.FUTURES,
                    event_name=mkt.get("title", ""),
                    outcome="YES",
                    price=yes_price,
                    source="kalshi",
                    timestamp=now,
                    raw_price=mkt.get("yes_ask"),
                    volume=mkt.get("volume"),
                    metadata={"ticker": mkt.get("ticker")},
                )
            )
        return markets

    def fetch_polymarket_markets(self, sport: Sport) -> list[MarketPrice]:
        """Fetch active sports markets from Polymarket CLOB API."""
        search_terms = {
            Sport.NFL: "NFL",
            Sport.NBA: "NBA",
            Sport.MLB: "MLB",
        }

        try:
            resp = self._client.get(
                "https://clob.polymarket.com/markets",
                params={"next_cursor": "MA=="},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            logger.warning("Polymarket API error: %s", e)
            return []

        markets = []
        now = datetime.utcnow()
        search = search_terms.get(sport, "").lower()

        for mkt in data.get("data", data) if isinstance(data, dict) else data:
            question = mkt.get("question", "")
            if search not in question.lower():
                continue

            for token in mkt.get("tokens", []):
                price = float(token.get("price", 0))
                markets.append(
                    MarketPrice(
                        sport=sport,
                        market_type=MarketType.FUTURES,
                        event_name=question,
                        outcome=token.get("outcome", ""),
                        price=price,
                        source="polymarket",
                        timestamp=now,
                        volume=mkt.get("volume"),
                        metadata={
                            "condition_id": mkt.get("condition_id"),
                            "token_id": token.get("token_id"),
                        },
                    )
                )
        return markets

    def get_all_market_prices(self, sport: Sport) -> pd.DataFrame:
        """Fetch and combine market prices from all sources."""
        all_prices: list[MarketPrice] = []
        all_prices.extend(self.fetch_kalshi_markets(sport))
        all_prices.extend(self.fetch_polymarket_markets(sport))

        if not all_prices:
            return pd.DataFrame()

        records = [
            {
                "sport": p.sport.value,
                "market_type": p.market_type.value,
                "event_name": p.event_name,
                "outcome": p.outcome,
                "price": p.price,
                "source": p.source,
                "timestamp": p.timestamp,
                "volume": p.volume,
            }
            for p in all_prices
        ]
        return pd.DataFrame(records)

    def build_price_history(
        self, sport: Sport, lookback_hours: int = 24
    ) -> pd.DataFrame:
        """Build time series of market prices for line movement tracking."""
        cache_key = f"market_history_{sport.value}"

        # Load existing history
        existing = None
        if self.cache:
            existing = self.cache.get_df(cache_key)

        current = self.get_all_market_prices(sport)

        if existing is not None and not current.empty:
            combined = pd.concat([existing, current], ignore_index=True)
            # Deduplicate and keep latest
            combined = combined.drop_duplicates(
                subset=["event_name", "outcome", "source", "timestamp"]
            )
            cutoff = datetime.utcnow().timestamp() - lookback_hours * 3600
            if "timestamp" in combined.columns:
                combined = combined[
                    combined["timestamp"].apply(
                        lambda t: t.timestamp() if hasattr(t, "timestamp") else 0
                    )
                    > cutoff
                ]
        elif not current.empty:
            combined = current
        else:
            combined = existing if existing is not None else pd.DataFrame()

        if self.cache and not combined.empty:
            self.cache.set_df(cache_key, combined)

        return combined

    def close(self):
        """Close HTTP client."""
        self._client.close()

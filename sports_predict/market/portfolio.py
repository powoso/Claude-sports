"""Portfolio management for prediction market positions.

Tracks positions, P&L, and enforces risk limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """A single market position."""

    event_name: str
    outcome: str
    source: str  # 'kalshi', 'polymarket'
    entry_price: float
    current_price: float
    size: float  # in dollars
    side: str  # 'yes' / 'no'
    entry_time: datetime
    sport: str
    edge_at_entry: float
    resolved: bool = False
    result: float | None = None  # 1.0 for win, 0.0 for loss

    @property
    def unrealized_pnl(self) -> float:
        if self.side == "yes":
            return self.size * (self.current_price - self.entry_price) / self.entry_price
        else:
            return self.size * (self.entry_price - self.current_price) / (1 - self.entry_price)

    @property
    def realized_pnl(self) -> float | None:
        if self.result is None:
            return None
        if self.side == "yes":
            return self.size * (self.result - self.entry_price) / self.entry_price
        else:
            return self.size * ((1 - self.result) - (1 - self.entry_price)) / (1 - self.entry_price)


class PortfolioManager:
    """Manages a portfolio of prediction market positions."""

    def __init__(
        self,
        bankroll: float = 10000.0,
        max_position_pct: float = 5.0,
        max_sport_exposure_pct: float = 30.0,
        max_correlated_exposure_pct: float = 15.0,
    ):
        self.bankroll = bankroll
        self.max_position_pct = max_position_pct
        self.max_sport_exposure_pct = max_sport_exposure_pct
        self.max_correlated_exposure_pct = max_correlated_exposure_pct
        self.positions: list[Position] = []
        self.closed_positions: list[Position] = []

    def current_exposure(self) -> float:
        """Total current exposure across all positions."""
        return sum(p.size for p in self.positions if not p.resolved)

    def sport_exposure(self, sport: str) -> float:
        """Current exposure for a specific sport."""
        return sum(
            p.size for p in self.positions
            if not p.resolved and p.sport == sport
        )

    def can_add_position(self, sport: str, size: float) -> tuple[bool, str]:
        """Check if a new position passes risk checks."""
        # Total exposure check
        total_after = self.current_exposure() + size
        if total_after > self.bankroll:
            return False, f"Total exposure ({total_after:.0f}) would exceed bankroll ({self.bankroll:.0f})"

        # Position size check
        if size > self.bankroll * self.max_position_pct / 100:
            return False, f"Position size ({size:.0f}) exceeds {self.max_position_pct}% max"

        # Sport exposure check
        sport_after = self.sport_exposure(sport) + size
        max_sport = self.bankroll * self.max_sport_exposure_pct / 100
        if sport_after > max_sport:
            return False, f"{sport} exposure ({sport_after:.0f}) would exceed {self.max_sport_exposure_pct}% max ({max_sport:.0f})"

        return True, "OK"

    def add_position(
        self,
        event_name: str,
        outcome: str,
        source: str,
        entry_price: float,
        size_pct: float,
        side: str,
        sport: str,
        edge: float,
    ) -> Position | None:
        """Add a new position to the portfolio.

        Args:
            size_pct: Position size as percentage of bankroll

        Returns:
            Position if added, None if rejected by risk checks
        """
        size = self.bankroll * size_pct / 100

        ok, reason = self.can_add_position(sport, size)
        if not ok:
            logger.warning("Position rejected: %s", reason)
            return None

        pos = Position(
            event_name=event_name,
            outcome=outcome,
            source=source,
            entry_price=entry_price,
            current_price=entry_price,
            size=size,
            side=side,
            entry_time=datetime.utcnow(),
            sport=sport,
            edge_at_entry=edge,
        )
        self.positions.append(pos)
        logger.info("Added position: %s %s @ %.2f (%.0f)", event_name, side, entry_price, size)
        return pos

    def update_prices(self, price_updates: dict[str, float]) -> None:
        """Update current prices for all positions.

        Args:
            price_updates: Dict of event_name -> current_price
        """
        for pos in self.positions:
            if pos.event_name in price_updates and not pos.resolved:
                pos.current_price = price_updates[pos.event_name]

    def resolve_position(self, event_name: str, result: float) -> None:
        """Resolve a position (event completed).

        Args:
            result: 1.0 for YES outcome, 0.0 for NO outcome
        """
        for pos in self.positions:
            if pos.event_name == event_name and not pos.resolved:
                pos.resolved = True
                pos.result = result
                self.closed_positions.append(pos)
                pnl = pos.realized_pnl
                logger.info(
                    "Resolved: %s → %s (PnL: $%.2f)",
                    event_name, "WIN" if result == 1.0 else "LOSS", pnl or 0,
                )

        # Remove resolved from active
        self.positions = [p for p in self.positions if not p.resolved]

    def get_summary(self) -> dict:
        """Get portfolio summary statistics."""
        active = [p for p in self.positions if not p.resolved]
        total_exposure = sum(p.size for p in active)
        unrealized_pnl = sum(p.unrealized_pnl for p in active)
        realized_pnl = sum(
            p.realized_pnl for p in self.closed_positions if p.realized_pnl is not None
        )

        # Win rate
        resolved = [p for p in self.closed_positions if p.result is not None]
        wins = sum(
            1 for p in resolved
            if (p.side == "yes" and p.result == 1.0)
            or (p.side == "no" and p.result == 0.0)
        )
        win_rate = wins / len(resolved) if resolved else 0.0

        # By sport
        sport_summary = {}
        for sport in set(p.sport for p in active):
            sport_positions = [p for p in active if p.sport == sport]
            sport_summary[sport] = {
                "positions": len(sport_positions),
                "exposure": sum(p.size for p in sport_positions),
                "unrealized_pnl": sum(p.unrealized_pnl for p in sport_positions),
            }

        return {
            "bankroll": self.bankroll,
            "active_positions": len(active),
            "total_exposure": round(total_exposure, 2),
            "exposure_pct": round(total_exposure / self.bankroll * 100, 1),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "realized_pnl": round(realized_pnl, 2),
            "total_pnl": round(unrealized_pnl + realized_pnl, 2),
            "roi_pct": round(
                (unrealized_pnl + realized_pnl) / self.bankroll * 100, 2
            ),
            "win_rate": round(win_rate, 3),
            "total_resolved": len(resolved),
            "by_sport": sport_summary,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Export all positions as a DataFrame."""
        records = []
        for p in self.positions + self.closed_positions:
            records.append({
                "event_name": p.event_name,
                "outcome": p.outcome,
                "source": p.source,
                "side": p.side,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "size": p.size,
                "sport": p.sport,
                "edge_at_entry": p.edge_at_entry,
                "entry_time": p.entry_time,
                "resolved": p.resolved,
                "result": p.result,
                "unrealized_pnl": p.unrealized_pnl,
                "realized_pnl": p.realized_pnl,
            })
        return pd.DataFrame(records)

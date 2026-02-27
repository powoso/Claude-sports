"""Tests for portfolio management."""

import pytest

from sports_predict.market.portfolio import PortfolioManager, Position


class TestPortfolioManager:
    def setup_method(self):
        self.pm = PortfolioManager(
            bankroll=10000.0,
            max_position_pct=5.0,
            max_sport_exposure_pct=30.0,
        )

    def test_add_position(self):
        pos = self.pm.add_position(
            event_name="Game1",
            outcome="YES",
            source="kalshi",
            entry_price=0.50,
            size_pct=2.0,
            side="yes",
            sport="nfl",
            edge=5.0,
        )
        assert pos is not None
        assert pos.size == 200.0
        assert len(self.pm.positions) == 1

    def test_position_size_limit(self):
        pos = self.pm.add_position(
            event_name="Game1",
            outcome="YES",
            source="kalshi",
            entry_price=0.50,
            size_pct=10.0,  # exceeds 5% max
            side="yes",
            sport="nfl",
            edge=5.0,
        )
        assert pos is None

    def test_sport_exposure_limit(self):
        # Add positions up to sport limit
        for i in range(6):
            self.pm.add_position(
                event_name=f"Game{i}",
                outcome="YES",
                source="kalshi",
                entry_price=0.50,
                size_pct=5.0,
                side="yes",
                sport="nfl",
                edge=5.0,
            )

        # 6 positions * 5% = 30%, at the limit
        # Next position should be rejected
        pos = self.pm.add_position(
            event_name="Game7",
            outcome="YES",
            source="kalshi",
            entry_price=0.50,
            size_pct=2.0,
            side="yes",
            sport="nfl",
            edge=5.0,
        )
        assert pos is None

    def test_resolve_position(self):
        self.pm.add_position(
            event_name="Game1",
            outcome="YES",
            source="kalshi",
            entry_price=0.50,
            size_pct=2.0,
            side="yes",
            sport="nfl",
            edge=5.0,
        )
        self.pm.resolve_position("Game1", 1.0)  # WIN

        assert len(self.pm.positions) == 0
        assert len(self.pm.closed_positions) == 1
        assert self.pm.closed_positions[0].result == 1.0

    def test_unrealized_pnl(self):
        self.pm.add_position(
            event_name="Game1",
            outcome="YES",
            source="kalshi",
            entry_price=0.50,
            size_pct=2.0,
            side="yes",
            sport="nfl",
            edge=5.0,
        )
        # Price moves up
        self.pm.update_prices({"Game1": 0.60})
        pnl = self.pm.positions[0].unrealized_pnl
        assert pnl > 0

    def test_portfolio_summary(self):
        self.pm.add_position(
            event_name="Game1",
            outcome="YES",
            source="kalshi",
            entry_price=0.50,
            size_pct=2.0,
            side="yes",
            sport="nfl",
            edge=5.0,
        )

        summary = self.pm.get_summary()
        assert summary["bankroll"] == 10000.0
        assert summary["active_positions"] == 1
        assert summary["total_exposure"] == 200.0
        assert summary["exposure_pct"] == 2.0

    def test_to_dataframe(self):
        self.pm.add_position(
            event_name="Game1",
            outcome="YES",
            source="kalshi",
            entry_price=0.50,
            size_pct=2.0,
            side="yes",
            sport="nfl",
            edge=5.0,
        )
        df = self.pm.to_dataframe()
        assert len(df) == 1
        assert "event_name" in df.columns
        assert "entry_price" in df.columns

    def test_win_rate_tracking(self):
        for i in range(10):
            self.pm.add_position(
                event_name=f"Game{i}",
                outcome="YES",
                source="kalshi",
                entry_price=0.50,
                size_pct=1.0,
                side="yes",
                sport="nfl",
                edge=5.0,
            )
            # 7 wins, 3 losses
            self.pm.resolve_position(f"Game{i}", 1.0 if i < 7 else 0.0)

        summary = self.pm.get_summary()
        assert summary["win_rate"] == pytest.approx(0.7)
        assert summary["total_resolved"] == 10


class TestPosition:
    def test_unrealized_pnl_yes_up(self):
        pos = Position(
            event_name="Test", outcome="YES", source="kalshi",
            entry_price=0.50, current_price=0.60, size=100.0,
            side="yes", entry_time=None, sport="nfl", edge_at_entry=5.0,
        )
        assert pos.unrealized_pnl > 0

    def test_unrealized_pnl_yes_down(self):
        pos = Position(
            event_name="Test", outcome="YES", source="kalshi",
            entry_price=0.50, current_price=0.40, size=100.0,
            side="yes", entry_time=None, sport="nfl", edge_at_entry=5.0,
        )
        assert pos.unrealized_pnl < 0

    def test_realized_pnl_win(self):
        pos = Position(
            event_name="Test", outcome="YES", source="kalshi",
            entry_price=0.50, current_price=0.50, size=100.0,
            side="yes", entry_time=None, sport="nfl", edge_at_entry=5.0,
            resolved=True, result=1.0,
        )
        assert pos.realized_pnl == pytest.approx(100.0)

    def test_realized_pnl_loss(self):
        pos = Position(
            event_name="Test", outcome="YES", source="kalshi",
            entry_price=0.50, current_price=0.50, size=100.0,
            side="yes", entry_time=None, sport="nfl", edge_at_entry=5.0,
            resolved=True, result=0.0,
        )
        assert pos.realized_pnl == pytest.approx(-100.0)

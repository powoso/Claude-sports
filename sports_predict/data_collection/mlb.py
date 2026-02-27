"""MLB data collection using pybaseball for Statcast, game logs, and park factors."""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sports_predict.utils.cache import DataCache

logger = logging.getLogger(__name__)

# Park factors (runs scored relative to average, 100 = average)
# Source: ESPN Park Factors, approximate values
PARK_FACTORS: dict[str, float] = {
    "COL": 1.30, "CIN": 1.08, "TEX": 1.06, "BOS": 1.05, "CHC": 1.04,
    "PHI": 1.03, "MIL": 1.02, "ATL": 1.02, "BAL": 1.01, "MIN": 1.01,
    "WSH": 1.00, "ARI": 1.00, "CLE": 0.99, "DET": 0.99, "LAA": 0.99,
    "NYY": 0.98, "TOR": 0.98, "STL": 0.98, "KC": 0.97, "CHW": 0.97,
    "HOU": 0.96, "SD": 0.96, "PIT": 0.96, "SF": 0.95, "TB": 0.95,
    "SEA": 0.94, "LAD": 0.94, "NYM": 0.94, "OAK": 0.93, "MIA": 0.92,
}


class MLBDataCollector:
    """Collects MLB Statcast, game-level, and pitching data."""

    def __init__(self, seasons: list[int], cache: DataCache | None = None):
        self.seasons = seasons
        self.cache = cache

    def get_statcast_batter(self, season: int) -> pd.DataFrame:
        """Fetch Statcast batting data for a season."""
        cache_key = f"mlb_statcast_batter_{season}"
        if self.cache:
            cached = self.cache.get_df(cache_key)
            if cached is not None:
                return cached

        from pybaseball import statcast_batter_exitvelo_barrels

        logger.info("Fetching MLB Statcast batter data for %d", season)
        df = statcast_batter_exitvelo_barrels(season)

        if self.cache and df is not None and not df.empty:
            self.cache.set_df(cache_key, df)
        return df if df is not None else pd.DataFrame()

    def get_statcast_pitcher(self, season: int) -> pd.DataFrame:
        """Fetch Statcast pitching data for a season."""
        cache_key = f"mlb_statcast_pitcher_{season}"
        if self.cache:
            cached = self.cache.get_df(cache_key)
            if cached is not None:
                return cached

        from pybaseball import statcast_pitcher_exitvelo_barrels

        logger.info("Fetching MLB Statcast pitcher data for %d", season)
        df = statcast_pitcher_exitvelo_barrels(season)

        if self.cache and df is not None and not df.empty:
            self.cache.set_df(cache_key, df)
        return df if df is not None else pd.DataFrame()

    def get_team_batting(self, season: int) -> pd.DataFrame:
        """Fetch team-level batting statistics."""
        cache_key = f"mlb_team_batting_{season}"
        if self.cache:
            cached = self.cache.get_df(cache_key)
            if cached is not None:
                return cached

        from pybaseball import team_batting

        logger.info("Fetching MLB team batting for %d", season)
        df = team_batting(season)
        df["Season"] = season

        if self.cache:
            self.cache.set_df(cache_key, df)
        return df

    def get_team_pitching(self, season: int) -> pd.DataFrame:
        """Fetch team-level pitching statistics."""
        cache_key = f"mlb_team_pitching_{season}"
        if self.cache:
            cached = self.cache.get_df(cache_key)
            if cached is not None:
                return cached

        from pybaseball import team_pitching

        logger.info("Fetching MLB team pitching for %d", season)
        df = team_pitching(season)
        df["Season"] = season

        if self.cache:
            self.cache.set_df(cache_key, df)
        return df

    def get_standings(self, season: int) -> pd.DataFrame:
        """Fetch season standings."""
        cache_key = f"mlb_standings_{season}"
        if self.cache:
            cached = self.cache.get_df(cache_key)
            if cached is not None:
                return cached

        from pybaseball import standings

        logger.info("Fetching MLB standings for %d", season)
        tables = standings(season)
        # standings returns a list of DataFrames (one per division)
        df = pd.concat(tables, ignore_index=True)
        df["Season"] = season

        if self.cache:
            self.cache.set_df(cache_key, df)
        return df

    def get_pitching_stats(self, season: int, min_ip: int = 30) -> pd.DataFrame:
        """Fetch individual pitcher stats via FanGraphs."""
        cache_key = f"mlb_pitching_stats_{season}_{min_ip}"
        if self.cache:
            cached = self.cache.get_df(cache_key)
            if cached is not None:
                return cached

        from pybaseball import pitching_stats

        logger.info("Fetching MLB pitching stats for %d (min IP=%d)", season, min_ip)
        df = pitching_stats(season, qual=min_ip)

        if self.cache and df is not None and not df.empty:
            self.cache.set_df(cache_key, df)
        return df if df is not None else pd.DataFrame()

    def get_batting_stats(self, season: int, min_pa: int = 100) -> pd.DataFrame:
        """Fetch individual batter stats via FanGraphs."""
        cache_key = f"mlb_batting_stats_{season}_{min_pa}"
        if self.cache:
            cached = self.cache.get_df(cache_key)
            if cached is not None:
                return cached

        from pybaseball import batting_stats

        logger.info("Fetching MLB batting stats for %d (min PA=%d)", season, min_pa)
        df = batting_stats(season, qual=min_pa)

        if self.cache and df is not None and not df.empty:
            self.cache.set_df(cache_key, df)
        return df if df is not None else pd.DataFrame()

    def compute_team_xwoba(self, season: int) -> pd.DataFrame:
        """Compute team-level expected weighted on-base average (xwOBA).

        Uses batting stats to approximate xwOBA from available metrics.
        """
        batting = self.get_team_batting(season)

        if batting.empty:
            return pd.DataFrame()

        # Use available OBP/SLG data to construct weighted metric
        # xwOBA weights: BB=0.69, HBP=0.72, 1B=0.88, 2B=1.24, 3B=1.56, HR=2.01
        xwoba_weights = {
            "BB_w": 0.69, "HBP_w": 0.72, "1B_w": 0.88,
            "2B_w": 1.24, "3B_w": 1.56, "HR_w": 2.01,
        }

        result = batting.copy()

        # Calculate singles from hits
        for col in ["H", "2B", "3B", "HR", "BB", "HBP", "AB", "SF"]:
            if col not in result.columns:
                result[col] = 0

        result["1B"] = result["H"] - result["2B"] - result["3B"] - result["HR"]
        result["PA"] = result["AB"] + result["BB"] + result["HBP"] + result["SF"]

        # Compute wOBA
        numerator = (
            xwoba_weights["BB_w"] * result["BB"]
            + xwoba_weights["HBP_w"] * result["HBP"]
            + xwoba_weights["1B_w"] * result["1B"]
            + xwoba_weights["2B_w"] * result["2B"]
            + xwoba_weights["3B_w"] * result["3B"]
            + xwoba_weights["HR_w"] * result["HR"]
        )
        result["wOBA"] = np.where(result["PA"] > 0, numerator / result["PA"], 0.0)

        keep = ["Team", "Season", "PA", "wOBA", "H", "HR", "BB", "1B", "2B", "3B"]
        available = [c for c in keep if c in result.columns]
        return result[available].copy()

    @staticmethod
    def get_park_factor(team: str) -> float:
        """Get park factor for a team's home stadium."""
        return PARK_FACTORS.get(team, 1.0)

    def compute_platoon_splits(self, season: int) -> pd.DataFrame:
        """Compute handedness platoon splits from batting stats."""
        batting = self.get_batting_stats(season, min_pa=50)

        if batting.empty:
            return pd.DataFrame()

        # pybaseball batting_stats includes Split column if requested
        # For now, aggregate team-level OPS as proxy
        if "Team" in batting.columns and "OPS" in batting.columns:
            splits = (
                batting.groupby("Team")
                .agg(avg_ops=("OPS", "mean"), players=("Name", "count"))
                .reset_index()
            )
            splits["season"] = season
            return splits

        return pd.DataFrame()

"""NFL data collection using nfl_data_py for play-by-play and seasonal data."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sports_predict.utils.cache import DataCache

logger = logging.getLogger(__name__)


class NFLDataCollector:
    """Collects NFL play-by-play, roster, schedule, and injury data."""

    def __init__(self, seasons: list[int], cache: DataCache | None = None):
        self.seasons = seasons
        self.cache = cache

    def get_pbp(self, season: int) -> pd.DataFrame:
        """Fetch play-by-play data for a given season."""
        cache_key = f"nfl_pbp_{season}"
        if self.cache:
            cached = self.cache.get_df(cache_key)
            if cached is not None:
                logger.info("Using cached NFL PBP for %d", season)
                return cached

        import nfl_data_py as nfl

        logger.info("Fetching NFL PBP for %d", season)
        df = nfl.import_pbp_data([season])

        # Keep essential columns to reduce memory
        keep_cols = [
            "game_id", "play_id", "season", "week", "game_date",
            "posteam", "defteam", "home_team", "away_team",
            "play_type", "yards_gained", "epa", "wpa", "wp", "def_wp",
            "qb_epa", "air_epa", "yac_epa", "comp_air_epa", "comp_yac_epa",
            "air_yards", "yards_after_catch",
            "pass_attempt", "rush_attempt", "complete_pass", "interception",
            "fumble_lost", "sack", "touchdown", "field_goal_attempt",
            "field_goal_result", "extra_point_attempt",
            "down", "ydstogo", "yardline_100",
            "score_differential", "half_seconds_remaining",
            "posteam_score", "defteam_score",
            "passer_player_name", "rusher_player_name", "receiver_player_name",
            "cp", "cpoe",
            "roof", "surface", "temp", "wind",
        ]
        available = [c for c in keep_cols if c in df.columns]
        df = df[available]

        if self.cache:
            self.cache.set_df(cache_key, df)
        return df

    def get_schedules(self, season: int) -> pd.DataFrame:
        """Fetch game schedule and results."""
        cache_key = f"nfl_schedule_{season}"
        if self.cache:
            cached = self.cache.get_df(cache_key)
            if cached is not None:
                return cached

        import nfl_data_py as nfl

        logger.info("Fetching NFL schedule for %d", season)
        df = nfl.import_schedules([season])

        if self.cache:
            self.cache.set_df(cache_key, df)
        return df

    def get_rosters(self, season: int) -> pd.DataFrame:
        """Fetch weekly roster data."""
        cache_key = f"nfl_rosters_{season}"
        if self.cache:
            cached = self.cache.get_df(cache_key)
            if cached is not None:
                return cached

        import nfl_data_py as nfl

        logger.info("Fetching NFL rosters for %d", season)
        df = nfl.import_weekly_rosters([season])

        if self.cache:
            self.cache.set_df(cache_key, df)
        return df

    def get_injuries(self, season: int) -> pd.DataFrame:
        """Fetch injury report data."""
        cache_key = f"nfl_injuries_{season}"
        if self.cache:
            cached = self.cache.get_df(cache_key)
            if cached is not None:
                return cached

        import nfl_data_py as nfl

        logger.info("Fetching NFL injuries for %d", season)
        df = nfl.import_injuries([season])

        if self.cache:
            self.cache.set_df(cache_key, df)
        return df

    def compute_team_epa(self, season: int) -> pd.DataFrame:
        """Compute per-team EPA metrics from play-by-play data.

        Returns DataFrame with columns:
            team, week, off_epa_play, def_epa_play, pass_epa_play,
            rush_epa_play, epa_play (net)
        """
        pbp = self.get_pbp(season)

        # Filter to actual plays (exclude penalties, timeouts, etc.)
        plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

        # Offensive EPA per play by team and week
        off_epa = (
            plays.groupby(["posteam", "week"])["epa"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "off_epa_play", "count": "off_plays"})
            .reset_index()
            .rename(columns={"posteam": "team"})
        )

        # Defensive EPA per play (lower is better for defense)
        def_epa = (
            plays.groupby(["defteam", "week"])["epa"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "def_epa_play", "count": "def_plays"})
            .reset_index()
            .rename(columns={"defteam": "team"})
        )

        # Pass EPA
        pass_plays = plays[plays["play_type"] == "pass"]
        pass_epa = (
            pass_plays.groupby(["posteam", "week"])["epa"]
            .mean()
            .reset_index()
            .rename(columns={"posteam": "team", "epa": "pass_epa_play"})
        )

        # Rush EPA
        rush_plays = plays[plays["play_type"] == "run"]
        rush_epa = (
            rush_plays.groupby(["posteam", "week"])["epa"]
            .mean()
            .reset_index()
            .rename(columns={"posteam": "team", "epa": "rush_epa_play"})
        )

        # Merge all
        result = off_epa.merge(def_epa, on=["team", "week"], how="outer")
        result = result.merge(pass_epa, on=["team", "week"], how="left")
        result = result.merge(rush_epa, on=["team", "week"], how="left")

        # Net EPA (offense - defense, since negative def EPA is good)
        result["epa_play"] = result["off_epa_play"] - result["def_epa_play"]

        # Fill NaN with 0 for teams with no pass/rush in a week
        for col in ["pass_epa_play", "rush_epa_play"]:
            result[col] = result[col].fillna(0.0)

        result["season"] = season
        return result.sort_values(["team", "week"]).reset_index(drop=True)

    def compute_advanced_passing(self, season: int) -> pd.DataFrame:
        """Compute advanced passing metrics: CPOE, air yards, pressure rate."""
        pbp = self.get_pbp(season)
        passes = pbp[pbp["play_type"] == "pass"].copy()

        qb_stats = (
            passes.groupby(["passer_player_name", "posteam", "week"])
            .agg(
                attempts=("pass_attempt", "sum"),
                completions=("complete_pass", "sum"),
                epa=("epa", "mean"),
                cpoe=("cpoe", "mean"),
                air_yards=("air_yards", "mean"),
                yac=("yards_after_catch", "mean"),
                sacks=("sack", "sum"),
                interceptions=("interception", "sum"),
            )
            .reset_index()
        )

        # Completion % and sack rate
        qb_stats["comp_pct"] = np.where(
            qb_stats["attempts"] > 0,
            qb_stats["completions"] / qb_stats["attempts"],
            0.0,
        )
        total_dropbacks = qb_stats["attempts"] + qb_stats["sacks"]
        qb_stats["sack_rate"] = np.where(
            total_dropbacks > 0,
            qb_stats["sacks"] / total_dropbacks,
            0.0,
        )

        qb_stats["season"] = season
        return qb_stats

    def compute_redzone_efficiency(self, season: int) -> pd.DataFrame:
        """Compute red zone scoring efficiency by team."""
        pbp = self.get_pbp(season)
        plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

        # Red zone = inside opponent's 20
        rz = plays[plays["yardline_100"] <= 20]

        rz_stats = (
            rz.groupby(["posteam", "week"])
            .agg(
                rz_plays=("play_id", "count"),
                rz_tds=("touchdown", "sum"),
                rz_epa=("epa", "mean"),
            )
            .reset_index()
            .rename(columns={"posteam": "team"})
        )
        rz_stats["rz_td_rate"] = np.where(
            rz_stats["rz_plays"] > 0,
            rz_stats["rz_tds"] / rz_stats["rz_plays"],
            0.0,
        )
        rz_stats["season"] = season
        return rz_stats

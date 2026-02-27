"""NBA data collection using nba_api for game logs, tracking, and player stats."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sports_predict.utils.cache import DataCache

logger = logging.getLogger(__name__)

# NBA team abbreviation to full name mapping for travel distance
NBA_ARENAS: dict[str, tuple[float, float]] = {
    "ATL": (33.757, -84.396), "BOS": (42.366, -71.062), "BKN": (40.683, -73.975),
    "CHA": (35.225, -80.839), "CHI": (41.881, -87.674), "CLE": (41.496, -81.688),
    "DAL": (32.790, -96.810), "DEN": (39.749, -105.008), "DET": (42.341, -83.055),
    "GSW": (37.768, -122.388), "HOU": (29.751, -95.362), "IND": (39.764, -86.155),
    "LAC": (34.043, -118.267), "LAL": (34.043, -118.267), "MEM": (35.138, -90.051),
    "MIA": (25.781, -80.187), "MIL": (43.045, -87.917), "MIN": (44.980, -93.276),
    "NOP": (29.949, -90.082), "NYK": (40.751, -73.994), "OKC": (35.463, -97.515),
    "ORL": (28.539, -81.384), "PHI": (39.901, -75.172), "PHX": (33.446, -112.071),
    "POR": (45.532, -122.667), "SAC": (38.580, -121.500), "SAS": (29.427, -98.438),
    "TOR": (43.643, -79.379), "UTA": (40.768, -111.901), "WAS": (38.898, -77.021),
}


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in miles between two points on Earth."""
    R = 3959  # Earth's radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


class NBADataCollector:
    """Collects NBA game-level and player-level stats via nba_api."""

    def __init__(self, seasons: list[str], cache: DataCache | None = None):
        self.seasons = seasons
        self.cache = cache

    def get_team_game_logs(self, season: str) -> pd.DataFrame:
        """Fetch team game logs for a season (e.g., '2024-25')."""
        cache_key = f"nba_team_logs_{season}"
        if self.cache:
            cached = self.cache.get_df(cache_key)
            if cached is not None:
                return cached

        from nba_api.stats.endpoints import leaguegamelog

        logger.info("Fetching NBA team game logs for %s", season)
        log = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="T",
        )
        df = log.get_data_frames()[0]

        # Parse game date
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

        # Extract opponent and home/away
        df["HOME"] = df["MATCHUP"].str.contains("vs.")
        df["OPPONENT"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s*(\w+)")

        if self.cache:
            self.cache.set_df(cache_key, df)
        return df

    def get_player_game_logs(self, season: str) -> pd.DataFrame:
        """Fetch player-level game logs."""
        cache_key = f"nba_player_logs_{season}"
        if self.cache:
            cached = self.cache.get_df(cache_key)
            if cached is not None:
                return cached

        from nba_api.stats.endpoints import leaguegamelog

        logger.info("Fetching NBA player game logs for %s", season)
        log = leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="P",
        )
        df = log.get_data_frames()[0]
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

        if self.cache:
            self.cache.set_df(cache_key, df)
        return df

    def get_advanced_team_stats(self, season: str) -> pd.DataFrame:
        """Fetch advanced team stats (net rating, pace, etc.)."""
        cache_key = f"nba_advanced_team_{season}"
        if self.cache:
            cached = self.cache.get_df(cache_key)
            if cached is not None:
                return cached

        from nba_api.stats.endpoints import leaguedashteamstats

        logger.info("Fetching NBA advanced team stats for %s", season)
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
        )
        df = stats.get_data_frames()[0]

        if self.cache:
            self.cache.set_df(cache_key, df)
        return df

    def compute_team_net_rating(self, season: str) -> pd.DataFrame:
        """Compute rolling net rating for each team from game logs.

        Net Rating = Offensive Rating - Defensive Rating
        Uses per-game points/possessions to estimate.
        """
        logs = self.get_team_game_logs(season)

        # Estimate possessions: FGA - OREB + TOV + 0.44 * FTA
        logs["POSS"] = (
            logs["FGA"] - logs["OREB"] + logs["TOV"] + 0.44 * logs["FTA"]
        )

        # Points per 100 possessions (offensive rating proxy)
        logs["OFF_RTG"] = np.where(
            logs["POSS"] > 0, logs["PTS"] / logs["POSS"] * 100, 100.0
        )

        # Sort by team and date
        logs = logs.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

        # We need to pair games to get opponent stats for def rating
        # Use GAME_ID to match home/away
        home_games = logs[logs["HOME"]].set_index("GAME_ID")[["TEAM_ABBREVIATION", "PTS", "POSS"]]
        away_games = logs[~logs["HOME"]].set_index("GAME_ID")[["TEAM_ABBREVIATION", "PTS", "POSS"]]

        # For each game, merge opponent's points
        home_games = home_games.rename(columns={
            "TEAM_ABBREVIATION": "HOME_TEAM", "PTS": "HOME_PTS", "POSS": "HOME_POSS"
        })
        away_games = away_games.rename(columns={
            "TEAM_ABBREVIATION": "AWAY_TEAM", "PTS": "AWAY_PTS", "POSS": "AWAY_POSS"
        })
        paired = home_games.join(away_games, how="inner")

        # Build per-team records with opp points
        records = []
        for _, row in paired.iterrows():
            avg_poss = (row["HOME_POSS"] + row["AWAY_POSS"]) / 2
            if avg_poss > 0:
                records.append({
                    "team": row["HOME_TEAM"],
                    "pts": row["HOME_PTS"],
                    "opp_pts": row["AWAY_PTS"],
                    "poss": avg_poss,
                    "off_rtg": row["HOME_PTS"] / avg_poss * 100,
                    "def_rtg": row["AWAY_PTS"] / avg_poss * 100,
                })
                records.append({
                    "team": row["AWAY_TEAM"],
                    "pts": row["AWAY_PTS"],
                    "opp_pts": row["HOME_PTS"],
                    "poss": avg_poss,
                    "off_rtg": row["AWAY_PTS"] / avg_poss * 100,
                    "def_rtg": row["HOME_PTS"] / avg_poss * 100,
                })

        df = pd.DataFrame(records)
        df["net_rtg"] = df["off_rtg"] - df["def_rtg"]

        # Season-level aggregation
        team_ratings = (
            df.groupby("team")
            .agg(
                off_rtg=("off_rtg", "mean"),
                def_rtg=("def_rtg", "mean"),
                net_rtg=("net_rtg", "mean"),
                pace=("poss", "mean"),
                games=("pts", "count"),
            )
            .reset_index()
            .sort_values("net_rtg", ascending=False)
        )

        team_ratings["season"] = season
        return team_ratings

    def compute_rest_days(self, season: str) -> pd.DataFrame:
        """Compute rest days and back-to-back flags for each team game."""
        logs = self.get_team_game_logs(season)
        logs = logs.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

        rest_data = []
        for team, group in logs.groupby("TEAM_ABBREVIATION"):
            dates = group["GAME_DATE"].values
            for i, row in group.iterrows():
                if len(rest_data) == 0 or group.index.get_loc(i) == 0:
                    rest = 7  # assume rested for first game
                else:
                    idx = group.index.get_loc(i)
                    prev_date = dates[idx - 1]
                    rest = (row["GAME_DATE"] - pd.Timestamp(prev_date)).days

                rest_data.append({
                    "team": team,
                    "game_id": row["GAME_ID"],
                    "game_date": row["GAME_DATE"],
                    "rest_days": rest,
                    "is_b2b": rest <= 1,
                })

        return pd.DataFrame(rest_data)

    def compute_travel_miles(self, season: str) -> pd.DataFrame:
        """Estimate travel miles for each team between games."""
        logs = self.get_team_game_logs(season)
        logs = logs.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

        travel = []
        for team, group in logs.groupby("TEAM_ABBREVIATION"):
            prev_city = team  # Start at home
            for _, row in group.iterrows():
                # Determine city: home team or opponent
                if row["HOME"]:
                    current_city = team
                else:
                    current_city = row["OPPONENT"]

                if prev_city in NBA_ARENAS and current_city in NBA_ARENAS:
                    lat1, lon1 = NBA_ARENAS[prev_city]
                    lat2, lon2 = NBA_ARENAS[current_city]
                    miles = _haversine_miles(lat1, lon1, lat2, lon2)
                else:
                    miles = 0.0

                travel.append({
                    "team": team,
                    "game_id": row["GAME_ID"],
                    "game_date": row["GAME_DATE"],
                    "travel_miles": miles,
                })
                prev_city = current_city

        return pd.DataFrame(travel)

"""NFL feature engineering: EPA efficiency, situational factors, and game-level features."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sports_predict.data_collection.nfl import NFLDataCollector
from sports_predict.data_collection.weather import WeatherCollector

logger = logging.getLogger(__name__)


class NFLFeatureEngine:
    """Builds game-level feature vectors for NFL prediction models."""

    def __init__(
        self,
        collector: NFLDataCollector,
        weather: WeatherCollector | None = None,
        smoothing_window: int = 4,
    ):
        self.collector = collector
        self.weather = weather
        self.smoothing_window = smoothing_window

    def build_team_efficiency(self, season: int) -> pd.DataFrame:
        """Build rolling EPA-based efficiency metrics for each team-week.

        Returns DataFrame with smoothed offensive/defensive EPA metrics.
        """
        epa = self.collector.compute_team_epa(season)

        # Sort and compute rolling averages
        epa = epa.sort_values(["team", "week"]).reset_index(drop=True)
        window = self.smoothing_window

        smoothed = []
        for team, group in epa.groupby("team"):
            g = group.copy()
            for col in ["off_epa_play", "def_epa_play", "pass_epa_play",
                         "rush_epa_play", "epa_play"]:
                g[f"{col}_rolling"] = g[col].rolling(window, min_periods=1).mean()

            # Exponential weighted average for recency bias
            for col in ["off_epa_play", "def_epa_play"]:
                g[f"{col}_ewm"] = g[col].ewm(span=window, min_periods=1).mean()

            smoothed.append(g)

        return pd.concat(smoothed, ignore_index=True)

    def build_passing_features(self, season: int) -> pd.DataFrame:
        """Build QB-level passing features: CPOE, air yards, pressure rate."""
        qb_stats = self.collector.compute_advanced_passing(season)

        # Filter to QBs with meaningful attempts
        qb_stats = qb_stats[qb_stats["attempts"] >= 5].copy()

        # Rolling CPOE (more stable)
        qb_stats = qb_stats.sort_values(
            ["passer_player_name", "week"]
        ).reset_index(drop=True)

        smoothed = []
        for qb, group in qb_stats.groupby("passer_player_name"):
            g = group.copy()
            g["cpoe_rolling"] = g["cpoe"].rolling(
                self.smoothing_window, min_periods=1
            ).mean()
            g["epa_rolling"] = g["epa"].rolling(
                self.smoothing_window, min_periods=1
            ).mean()
            g["sack_rate_rolling"] = g["sack_rate"].rolling(
                self.smoothing_window, min_periods=1
            ).mean()
            smoothed.append(g)

        return pd.concat(smoothed, ignore_index=True)

    def build_redzone_features(self, season: int) -> pd.DataFrame:
        """Build red zone efficiency features."""
        rz = self.collector.compute_redzone_efficiency(season)
        rz = rz.sort_values(["team", "week"]).reset_index(drop=True)

        smoothed = []
        for team, group in rz.groupby("team"):
            g = group.copy()
            g["rz_td_rate_rolling"] = g["rz_td_rate"].rolling(
                self.smoothing_window, min_periods=1
            ).mean()
            g["rz_epa_rolling"] = g["rz_epa"].rolling(
                self.smoothing_window, min_periods=1
            ).mean()
            smoothed.append(g)

        return pd.concat(smoothed, ignore_index=True)

    def build_situational_features(self, season: int) -> pd.DataFrame:
        """Build situational features: rest days, travel, dome/outdoor, divisional."""
        schedule = self.collector.get_schedules(season)

        features = []
        for _, game in schedule.iterrows():
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            week = game.get("week", 0)

            feat = {
                "game_id": game.get("game_id", ""),
                "season": season,
                "week": week,
                "home_team": home,
                "away_team": away,
            }

            # Dome/outdoor
            roof = game.get("roof", "")
            feat["is_dome"] = 1 if roof in ("dome", "closed") else 0
            feat["is_outdoor"] = 1 if roof in ("outdoors", "open") else 0

            # Surface
            feat["is_turf"] = 1 if game.get("surface", "") == "fieldturf" else 0

            # Divisional game
            feat["is_divisional"] = 1 if game.get("div_game", 0) == 1 else 0

            # Home rest advantage
            home_rest = game.get("home_rest", 7)
            away_rest = game.get("away_rest", 7)
            feat["home_rest"] = home_rest
            feat["away_rest"] = away_rest
            feat["rest_advantage"] = home_rest - away_rest

            # Temperature and wind from schedule data (fallback)
            feat["temperature"] = game.get("temp", 65)
            feat["wind"] = game.get("wind", 5)

            # Spread and total (if available for calibration)
            feat["spread_line"] = game.get("spread_line", 0)
            feat["total_line"] = game.get("total_line", 45)

            # Result (for training)
            feat["home_score"] = game.get("home_score", np.nan)
            feat["away_score"] = game.get("away_score", np.nan)
            if not np.isnan(feat["home_score"]) and not np.isnan(feat["away_score"]):
                feat["home_win"] = 1 if feat["home_score"] > feat["away_score"] else 0
                feat["margin"] = feat["home_score"] - feat["away_score"]
            else:
                feat["home_win"] = np.nan
                feat["margin"] = np.nan

            features.append(feat)

        return pd.DataFrame(features)

    def build_game_features(self, season: int, week: int) -> pd.DataFrame:
        """Build complete feature vector for all games in a given week.

        Merges efficiency, passing, red zone, and situational features.
        """
        # Get all component features
        efficiency = self.build_team_efficiency(season)
        passing = self.build_passing_features(season)
        redzone = self.build_redzone_features(season)
        situational = self.build_situational_features(season)

        # Filter to the target week's games
        games = situational[situational["week"] == week].copy()

        if games.empty:
            return pd.DataFrame()

        # For prediction, use the PREVIOUS week's rolling stats
        prev_week = week - 1 if week > 1 else 1
        eff_prev = efficiency[efficiency["week"] == prev_week]

        # Merge home team efficiency
        home_eff = eff_prev.rename(columns=lambda c: f"home_{c}" if c not in ["team", "week", "season"] else c)
        games = games.merge(
            home_eff, left_on="home_team", right_on="team", how="left", suffixes=("", "_eff")
        )

        # Merge away team efficiency
        away_eff = eff_prev.rename(columns=lambda c: f"away_{c}" if c not in ["team", "week", "season"] else c)
        games = games.merge(
            away_eff, left_on="away_team", right_on="team", how="left", suffixes=("", "_away_eff")
        )

        # Compute differentials
        for metric in ["off_epa_play_rolling", "def_epa_play_rolling", "epa_play_rolling"]:
            home_col = f"home_{metric}"
            away_col = f"away_{metric}"
            if home_col in games.columns and away_col in games.columns:
                games[f"diff_{metric}"] = games[home_col] - games[away_col]

        # Drop redundant columns
        drop_cols = [c for c in games.columns if c.endswith("_eff")]
        games = games.drop(columns=drop_cols, errors="ignore")

        return games

    def compute_schedule_strength(self, season: int) -> pd.DataFrame:
        """Compute strength of schedule for each team based on opponent EPA."""
        epa = self.collector.compute_team_epa(season)

        # Season-level team quality
        team_quality = (
            epa.groupby("team")["epa_play"]
            .mean()
            .reset_index()
            .rename(columns={"epa_play": "team_quality"})
        )

        schedule = self.collector.get_schedules(season)

        sos_records = []
        for team in team_quality["team"].unique():
            # Find all opponents
            home_games = schedule[schedule["home_team"] == team]["away_team"]
            away_games = schedule[schedule["away_team"] == team]["home_team"]
            opponents = pd.concat([home_games, away_games])

            opp_quality = team_quality[
                team_quality["team"].isin(opponents)
            ]["team_quality"].mean()

            sos_records.append({
                "team": team,
                "sos": opp_quality,
                "games_played": len(opponents),
            })

        return pd.DataFrame(sos_records)

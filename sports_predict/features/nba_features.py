"""NBA feature engineering: net rating, pace-adjusted stats, rest/travel factors."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sports_predict.data_collection.nba import NBADataCollector

logger = logging.getLogger(__name__)


class NBAFeatureEngine:
    """Builds game-level feature vectors for NBA prediction models."""

    def __init__(
        self,
        collector: NBADataCollector,
        rolling_window: int = 10,
        rest_penalty_b2b: float = -2.5,
        travel_penalty_per_1000mi: float = -0.5,
    ):
        self.collector = collector
        self.rolling_window = rolling_window
        self.rest_penalty_b2b = rest_penalty_b2b
        self.travel_penalty_per_1000mi = travel_penalty_per_1000mi

    def build_rolling_ratings(self, season: str) -> pd.DataFrame:
        """Build rolling net rating for each team-game.

        Returns rolling offensive rating, defensive rating, and net rating.
        """
        logs = self.collector.get_team_game_logs(season)
        logs = logs.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

        # Estimate possessions
        logs["POSS"] = logs["FGA"] - logs["OREB"] + logs["TOV"] + 0.44 * logs["FTA"]
        logs["OFF_RTG"] = np.where(logs["POSS"] > 0, logs["PTS"] / logs["POSS"] * 100, 100.0)

        # Match games to get defensive rating
        home = logs[logs["HOME"]].set_index("GAME_ID")
        away = logs[~logs["HOME"]].set_index("GAME_ID")

        game_ratings = []
        for game_id in home.index.intersection(away.index):
            h = home.loc[game_id]
            a = away.loc[game_id]
            avg_poss = (h["POSS"] + a["POSS"]) / 2
            if avg_poss <= 0:
                continue

            # Home team
            game_ratings.append({
                "team": h["TEAM_ABBREVIATION"],
                "game_id": game_id,
                "game_date": h["GAME_DATE"],
                "off_rtg": h["PTS"] / avg_poss * 100,
                "def_rtg": a["PTS"] / avg_poss * 100,
                "net_rtg": (h["PTS"] - a["PTS"]) / avg_poss * 100,
                "pace": avg_poss,
                "pts": h["PTS"],
                "opp_pts": a["PTS"],
                "home": True,
                "win": h["WL"] == "W",
            })
            # Away team
            game_ratings.append({
                "team": a["TEAM_ABBREVIATION"],
                "game_id": game_id,
                "game_date": a["GAME_DATE"],
                "off_rtg": a["PTS"] / avg_poss * 100,
                "def_rtg": h["PTS"] / avg_poss * 100,
                "net_rtg": (a["PTS"] - h["PTS"]) / avg_poss * 100,
                "pace": avg_poss,
                "pts": a["PTS"],
                "opp_pts": h["PTS"],
                "home": False,
                "win": a["WL"] == "W",
            })

        df = pd.DataFrame(game_ratings)
        df = df.sort_values(["team", "game_date"]).reset_index(drop=True)

        # Rolling averages
        window = self.rolling_window
        smoothed = []
        for team, group in df.groupby("team"):
            g = group.copy()
            for col in ["off_rtg", "def_rtg", "net_rtg", "pace"]:
                g[f"{col}_rolling"] = g[col].rolling(window, min_periods=3).mean()
                g[f"{col}_ewm"] = g[col].ewm(span=window, min_periods=3).mean()
            smoothed.append(g)

        return pd.concat(smoothed, ignore_index=True)

    def build_rest_features(self, season: str) -> pd.DataFrame:
        """Build rest days, back-to-back flags, and rest advantage."""
        rest = self.collector.compute_rest_days(season)
        rest["rest_penalty"] = np.where(
            rest["is_b2b"], self.rest_penalty_b2b, 0.0
        )
        # 3-in-4 nights detection
        # (would need more context, approximated here)
        return rest

    def build_travel_features(self, season: str) -> pd.DataFrame:
        """Build travel distance features and fatigue adjustment."""
        travel = self.collector.compute_travel_miles(season)
        travel["travel_penalty"] = (
            travel["travel_miles"] / 1000.0 * self.travel_penalty_per_1000mi
        )
        return travel

    def build_home_away_splits(self, season: str) -> pd.DataFrame:
        """Compute home/away performance splits."""
        ratings = self.build_rolling_ratings(season)

        splits = (
            ratings.groupby(["team", "home"])
            .agg(
                net_rtg=("net_rtg", "mean"),
                off_rtg=("off_rtg", "mean"),
                def_rtg=("def_rtg", "mean"),
                win_pct=("win", "mean"),
                games=("game_id", "count"),
            )
            .reset_index()
        )

        # Pivot to get home/away in same row
        home_splits = splits[splits["home"]].drop(columns=["home"]).rename(
            columns=lambda c: f"home_{c}" if c != "team" else c
        )
        away_splits = splits[~splits["home"]].drop(columns=["home"]).rename(
            columns=lambda c: f"away_{c}" if c != "team" else c
        )

        merged = home_splits.merge(away_splits, on="team", how="outer")
        merged["home_court_advantage"] = (
            merged.get("home_net_rtg", 0) - merged.get("away_net_rtg", 0)
        )
        return merged

    def build_game_features(self, season: str, game_date: str | None = None) -> pd.DataFrame:
        """Build complete feature set for games on a given date.

        If game_date is None, builds features for all games in the season.
        """
        ratings = self.build_rolling_ratings(season)
        rest = self.build_rest_features(season)
        travel = self.build_travel_features(season)

        if game_date:
            target_date = pd.to_datetime(game_date)
            # Find games on this date
            target_games = ratings[
                ratings["game_date"].dt.date == target_date.date()
            ]
        else:
            target_games = ratings

        # Only take home team rows to avoid duplicates
        home_games = target_games[target_games["home"]].copy()

        if home_games.empty:
            return pd.DataFrame()

        # Merge away team ratings
        away_ratings = target_games[~target_games["home"]].copy()
        away_ratings = away_ratings.rename(
            columns=lambda c: f"away_{c}" if c not in ["game_id", "game_date"] else c
        )

        features = home_games.merge(
            away_ratings[["game_id", "away_team", "away_off_rtg_rolling",
                          "away_def_rtg_rolling", "away_net_rtg_rolling", "away_pace_rolling"]],
            on="game_id",
            how="left",
        )

        # Merge rest data
        home_rest = rest.rename(columns=lambda c: f"home_{c}" if c not in ["game_id"] else c)
        features = features.merge(home_rest[["game_id", "home_rest_days", "home_is_b2b"]],
                                   on="game_id", how="left")

        away_rest = rest.rename(columns=lambda c: f"away_{c}" if c not in ["game_id"] else c)
        features = features.merge(away_rest[["game_id", "away_rest_days", "away_is_b2b"]],
                                   on="game_id", how="left")

        # Compute differentials
        if "away_net_rtg_rolling" in features.columns:
            features["net_rtg_diff"] = (
                features.get("net_rtg_rolling", 0) - features.get("away_net_rtg_rolling", 0)
            )

        # Rest advantage
        features["rest_advantage"] = (
            features.get("home_rest_days", 3) - features.get("away_rest_days", 3)
        )

        return features

    def compute_clutch_stats(self, season: str) -> pd.DataFrame:
        """Compute clutch performance: games within 5 points in final 5 minutes."""
        logs = self.collector.get_team_game_logs(season)

        # Use final score margin as proxy for clutch (close games)
        logs = logs.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

        clutch = []
        for team, group in logs.groupby("TEAM_ABBREVIATION"):
            close_games = group[group["PLUS_MINUS"].abs() <= 5]
            total = len(group)
            close = len(close_games)
            close_wins = len(close_games[close_games["WL"] == "W"])

            clutch.append({
                "team": team,
                "close_games": close,
                "close_win_pct": close_wins / close if close > 0 else 0.5,
                "close_game_pct": close / total if total > 0 else 0.0,
                "total_games": total,
            })

        return pd.DataFrame(clutch)

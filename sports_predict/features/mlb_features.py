"""MLB feature engineering: xwOBA, park factors, platoon splits, bullpen fatigue."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sports_predict.data_collection.mlb import MLBDataCollector, PARK_FACTORS

logger = logging.getLogger(__name__)


class MLBFeatureEngine:
    """Builds game-level feature vectors for MLB prediction models."""

    def __init__(self, collector: MLBDataCollector, xwoba_smoothing_pa: int = 150):
        self.collector = collector
        self.xwoba_smoothing_pa = xwoba_smoothing_pa

    def build_team_offense(self, season: int) -> pd.DataFrame:
        """Build team-level offensive features: wOBA, ISO, BB%, K%."""
        batting = self.collector.get_team_batting(season)

        if batting.empty:
            return pd.DataFrame()

        result = batting.copy()

        # Compute singles
        for col in ["H", "2B", "3B", "HR", "BB", "HBP", "AB", "SO", "SF"]:
            if col not in result.columns:
                result[col] = 0

        result["1B"] = result["H"] - result["2B"] - result["3B"] - result["HR"]
        result["PA"] = result["AB"] + result["BB"] + result["HBP"] + result["SF"]

        # wOBA calculation
        weights = {"BB": 0.69, "HBP": 0.72, "1B": 0.88, "2B": 1.24, "3B": 1.56, "HR": 2.01}
        numerator = sum(weights[k] * result[k] for k in weights)
        result["wOBA"] = np.where(result["PA"] > 0, numerator / result["PA"], 0.0)

        # Isolated power
        result["ISO"] = result.get("SLG", 0) - result.get("AVG", 0)

        # Walk and strikeout rates
        result["BB_pct"] = np.where(result["PA"] > 0, result["BB"] / result["PA"], 0)
        result["K_pct"] = np.where(result["PA"] > 0, result["SO"] / result["PA"], 0)

        # Park-adjusted wOBA
        result["park_factor"] = result["Team"].map(
            lambda t: PARK_FACTORS.get(t, 1.0) if isinstance(t, str) else 1.0
        )
        result["wOBA_park_adj"] = result["wOBA"] / result["park_factor"]

        result["season"] = season
        return result

    def build_team_pitching(self, season: int) -> pd.DataFrame:
        """Build team-level pitching features: FIP, K/9, BB/9, HR/9."""
        pitching = self.collector.get_team_pitching(season)

        if pitching.empty:
            return pd.DataFrame()

        result = pitching.copy()

        # FIP calculation: (13*HR + 3*(BB+HBP) - 2*K) / IP + FIP_constant
        # FIP constant is approximately 3.10
        FIP_CONSTANT = 3.10
        for col in ["HR", "BB", "HBP", "SO", "IP"]:
            if col not in result.columns:
                result[col] = 0

        ip = result["IP"].replace(0, np.nan)
        result["FIP"] = (
            (13 * result["HR"] + 3 * (result["BB"] + result.get("HBP", 0)) - 2 * result["SO"])
            / ip
            + FIP_CONSTANT
        )

        # Rate stats per 9 innings
        result["K_per_9"] = np.where(ip > 0, result["SO"] / ip * 9, 0)
        result["BB_per_9"] = np.where(ip > 0, result["BB"] / ip * 9, 0)
        result["HR_per_9"] = np.where(ip > 0, result["HR"] / ip * 9, 0)

        # Park-adjusted ERA
        result["park_factor"] = result["Team"].map(
            lambda t: PARK_FACTORS.get(t, 1.0) if isinstance(t, str) else 1.0
        )
        era = result.get("ERA", pd.Series(dtype=float))
        result["ERA_park_adj"] = era / result["park_factor"]

        result["season"] = season
        return result

    def build_starter_features(self, season: int) -> pd.DataFrame:
        """Build starting pitcher features from individual stats."""
        pitchers = self.collector.get_pitching_stats(season, min_ip=30)

        if pitchers.empty:
            return pd.DataFrame()

        # Select key columns
        keep = ["Name", "Team", "W", "L", "ERA", "IP", "SO", "BB", "HR",
                "FIP", "WHIP", "WAR"]
        available = [c for c in keep if c in pitchers.columns]
        result = pitchers[available].copy()

        # Quality start rate proxy
        if "IP" in result.columns and "ERA" in result.columns:
            result["quality_proxy"] = np.where(
                result["ERA"] < 4.0, 1.0, 0.0
            ) * np.where(result["IP"] > 50, 1.0, 0.5)

        result["season"] = season
        return result

    def build_bullpen_features(self, season: int) -> pd.DataFrame:
        """Build bullpen quality and workload features."""
        pitchers = self.collector.get_pitching_stats(season, min_ip=10)

        if pitchers.empty:
            return pd.DataFrame()

        # Relievers typically have lower IP per appearance
        if "GS" in pitchers.columns:
            relievers = pitchers[pitchers["GS"] == 0].copy()
        else:
            relievers = pitchers[pitchers["IP"] < 80].copy()

        if relievers.empty or "Team" not in relievers.columns:
            return pd.DataFrame()

        # Aggregate by team
        bullpen = (
            relievers.groupby("Team")
            .agg(
                bp_era=("ERA", "mean"),
                bp_fip=("FIP", "mean") if "FIP" in relievers.columns else ("ERA", "mean"),
                bp_whip=("WHIP", "mean") if "WHIP" in relievers.columns else ("ERA", "mean"),
                bp_k_rate=("SO", "sum"),
                bp_bb=("BB", "sum"),
                bp_ip=("IP", "sum"),
                bp_pitchers=("Name", "count"),
            )
            .reset_index()
        )

        if bullpen["bp_ip"].sum() > 0:
            bullpen["bp_k_per_9"] = bullpen["bp_k_rate"] / bullpen["bp_ip"] * 9
            bullpen["bp_bb_per_9"] = bullpen["bp_bb"] / bullpen["bp_ip"] * 9

        bullpen["season"] = season
        return bullpen

    def build_game_features(
        self,
        season: int,
        home_team: str,
        away_team: str,
        home_starter: str | None = None,
        away_starter: str | None = None,
    ) -> dict:
        """Build complete feature dictionary for a single matchup.

        Returns a flat dictionary suitable for model input.
        """
        offense = self.build_team_offense(season)
        pitching = self.build_team_pitching(season)
        bullpen = self.build_bullpen_features(season)

        features: dict = {
            "home_team": home_team,
            "away_team": away_team,
            "season": season,
        }

        # Home team offense
        home_off = offense[offense["Team"] == home_team]
        if not home_off.empty:
            row = home_off.iloc[0]
            features["home_wOBA"] = row.get("wOBA", 0.32)
            features["home_wOBA_adj"] = row.get("wOBA_park_adj", 0.32)
            features["home_ISO"] = row.get("ISO", 0.15)
            features["home_BB_pct"] = row.get("BB_pct", 0.08)
            features["home_K_pct"] = row.get("K_pct", 0.22)

        # Away team offense
        away_off = offense[offense["Team"] == away_team]
        if not away_off.empty:
            row = away_off.iloc[0]
            features["away_wOBA"] = row.get("wOBA", 0.32)
            features["away_wOBA_adj"] = row.get("wOBA_park_adj", 0.32)
            features["away_ISO"] = row.get("ISO", 0.15)
            features["away_BB_pct"] = row.get("BB_pct", 0.08)
            features["away_K_pct"] = row.get("K_pct", 0.22)

        # Pitching
        home_pitch = pitching[pitching["Team"] == home_team]
        if not home_pitch.empty:
            row = home_pitch.iloc[0]
            features["home_FIP"] = row.get("FIP", 4.0)
            features["home_K_per_9"] = row.get("K_per_9", 8.0)
            features["home_ERA_adj"] = row.get("ERA_park_adj", 4.0)

        away_pitch = pitching[pitching["Team"] == away_team]
        if not away_pitch.empty:
            row = away_pitch.iloc[0]
            features["away_FIP"] = row.get("FIP", 4.0)
            features["away_K_per_9"] = row.get("K_per_9", 8.0)
            features["away_ERA_adj"] = row.get("ERA_park_adj", 4.0)

        # Bullpen
        if not bullpen.empty:
            home_bp = bullpen[bullpen["Team"] == home_team]
            if not home_bp.empty:
                features["home_bp_era"] = home_bp.iloc[0].get("bp_era", 4.0)
            away_bp = bullpen[bullpen["Team"] == away_team]
            if not away_bp.empty:
                features["away_bp_era"] = away_bp.iloc[0].get("bp_era", 4.0)

        # Park factor
        features["park_factor"] = PARK_FACTORS.get(home_team, 1.0)

        # Differentials
        features["wOBA_diff"] = features.get("home_wOBA", 0.32) - features.get("away_wOBA", 0.32)
        features["FIP_diff"] = features.get("away_FIP", 4.0) - features.get("home_FIP", 4.0)

        return features

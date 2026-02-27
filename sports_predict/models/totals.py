"""Game totals modeling: pace + efficiency → projected score distribution."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class TotalsPrediction:
    """Prediction for a game total (over/under)."""

    home_team: str
    away_team: str
    projected_total: float
    total_std: float
    line: float
    over_prob: float
    under_prob: float
    projected_home_score: float
    projected_away_score: float
    score_distribution: dict | None = None


class TotalsModel:
    """Projects game totals from team pace and efficiency metrics.

    For each sport:
    - NFL: possessions * points_per_possession for each team
    - NBA: pace * (off_rtg / 100) for each team
    - MLB: runs per game adjusted for park factor and pitching matchup
    """

    def __init__(self, sport: str = "nfl"):
        self.sport = sport
        self._league_avg = {
            "nfl": {"total": 45.0, "std": 10.0, "possessions": 11.5},
            "nba": {"total": 225.0, "std": 15.0, "pace": 100.0},
            "mlb": {"total": 8.5, "std": 3.0, "runs_per_game": 4.25},
        }.get(sport, {"total": 45.0, "std": 10.0})

    def predict_nfl_total(
        self,
        home_off_epa: float,
        home_def_epa: float,
        away_off_epa: float,
        away_def_epa: float,
        home_pace: float | None = None,
        away_pace: float | None = None,
        weather_factor: float = 1.0,
        dome: bool = False,
    ) -> TotalsPrediction:
        """Predict NFL game total from EPA metrics.

        EPA is converted to expected points:
        - Average NFL drive is ~6 plays
        - ~11-12 possessions per team per game
        - Points per drive ≈ (EPA/play * plays_per_drive) + base_scoring_rate
        """
        avg_possessions = home_pace or self._league_avg.get("possessions", 11.5)

        # Base scoring: ~2.1 points per drive in modern NFL
        base_ppd = 2.1

        # EPA adjustment: each 0.1 EPA/play ≈ 0.6 points/drive
        home_scoring_vs_away_def = base_ppd + (home_off_epa + away_def_epa) * 6.0
        away_scoring_vs_home_def = base_ppd + (away_off_epa + home_def_epa) * 6.0

        # Scale to points per game
        home_points = max(home_scoring_vs_away_def * avg_possessions, 7.0)
        away_points = max(away_scoring_vs_home_def * avg_possessions, 7.0)

        # Weather adjustment (wind/cold reduce scoring)
        if not dome:
            home_points *= weather_factor
            away_points *= weather_factor

        total = home_points + away_points
        std = self._league_avg.get("std", 10.0)

        return TotalsPrediction(
            home_team="",
            away_team="",
            projected_total=round(total, 1),
            total_std=std,
            line=0,  # Set externally
            over_prob=0.5,
            under_prob=0.5,
            projected_home_score=round(home_points, 1),
            projected_away_score=round(away_points, 1),
        )

    def predict_nba_total(
        self,
        home_off_rtg: float,
        home_def_rtg: float,
        away_off_rtg: float,
        away_def_rtg: float,
        home_pace: float,
        away_pace: float,
        league_avg_off_rtg: float = 112.0,
    ) -> TotalsPrediction:
        """Predict NBA game total from offensive/defensive ratings and pace.

        Total = expected_pace * (home_off_rtg_adj + away_off_rtg_adj) / 100
        """
        # Expected game pace (geometric mean of both teams' pace preferences)
        expected_pace = np.sqrt(home_pace * away_pace)

        # Opponent-adjusted ratings
        # Home offense vs away defense
        home_pts_per_100 = home_off_rtg + (away_def_rtg - league_avg_off_rtg) / 2
        # Away offense vs home defense
        away_pts_per_100 = away_off_rtg + (home_def_rtg - league_avg_off_rtg) / 2

        home_points = home_pts_per_100 * expected_pace / 100
        away_points = away_pts_per_100 * expected_pace / 100

        total = home_points + away_points
        std = self._league_avg.get("std", 15.0)

        return TotalsPrediction(
            home_team="",
            away_team="",
            projected_total=round(total, 1),
            total_std=std,
            line=0,
            over_prob=0.5,
            under_prob=0.5,
            projected_home_score=round(home_points, 1),
            projected_away_score=round(away_points, 1),
        )

    def predict_mlb_total(
        self,
        home_woba: float,
        away_woba: float,
        home_era: float,
        away_era: float,
        park_factor: float = 1.0,
        home_bullpen_era: float = 4.0,
        away_bullpen_era: float = 4.0,
    ) -> TotalsPrediction:
        """Predict MLB game total from batting and pitching metrics.

        Uses wOBA to estimate run production and ERA for pitching quality.
        """
        # wOBA to runs per game approximation
        # League avg wOBA ~.320 → ~4.5 R/G
        # Each .010 wOBA ≈ 0.14 R/G
        league_woba = 0.320
        league_rpg = 4.25

        home_run_rate = league_rpg + (home_woba - league_woba) / 0.010 * 0.14
        away_run_rate = league_rpg + (away_woba - league_woba) / 0.010 * 0.14

        # Pitching adjustment: ERA relative to league average (4.25 ERA)
        league_era = 4.25
        # Away starter faces home batting
        home_scoring = home_run_rate * (away_era / league_era)
        away_scoring = away_run_rate * (home_era / league_era)

        # Bullpen factor (weight: ~35% of game is bullpen)
        bp_weight = 0.35
        home_scoring = home_scoring * (1 - bp_weight) + (
            home_run_rate * (away_bullpen_era / league_era) * bp_weight
        )
        away_scoring = away_scoring * (1 - bp_weight) + (
            away_run_rate * (home_bullpen_era / league_era) * bp_weight
        )

        # Park factor
        home_scoring *= park_factor
        away_scoring *= park_factor

        total = max(home_scoring + away_scoring, 4.0)
        std = self._league_avg.get("std", 3.0)

        return TotalsPrediction(
            home_team="",
            away_team="",
            projected_total=round(total, 1),
            total_std=std,
            line=0,
            over_prob=0.5,
            under_prob=0.5,
            projected_home_score=round(home_scoring, 1),
            projected_away_score=round(away_scoring, 1),
        )

    def set_line_and_compute(
        self, prediction: TotalsPrediction, line: float
    ) -> TotalsPrediction:
        """Set the market line and compute over/under probabilities."""
        prediction.line = line

        # Use normal distribution for probability
        z = (line - prediction.projected_total) / prediction.total_std
        prediction.under_prob = float(sp_stats.norm.cdf(z))
        prediction.over_prob = 1.0 - prediction.under_prob

        return prediction

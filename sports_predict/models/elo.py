"""Elo rating system with sport-specific adjustments.

Implements a regularized Elo system with:
- Home-field/court advantage
- Margin of victory adjustment
- Season regression to mean
- Sport-specific K factors
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EloRating:
    """Single team's Elo state."""

    team: str
    rating: float
    games_played: int = 0
    wins: int = 0
    losses: int = 0


@dataclass
class EloConfig:
    """Elo system configuration."""

    k_factor: float = 20.0
    home_advantage: float = 48.0  # Elo points
    mean_rating: float = 1500.0
    initial_rating: float = 1500.0
    season_reversion: float = 0.33  # fraction to revert to mean between seasons
    mov_multiplier: float = 1.0  # margin of victory scaling


class EloModel:
    """Elo rating model with margin-of-victory adjustment."""

    def __init__(self, config: EloConfig | None = None):
        self.config = config or EloConfig()
        self.ratings: dict[str, EloRating] = {}
        self.history: list[dict] = []

    def _get_or_create(self, team: str) -> EloRating:
        if team not in self.ratings:
            self.ratings[team] = EloRating(
                team=team, rating=self.config.initial_rating
            )
        return self.ratings[team]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score (win probability) for team A."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def win_probability(
        self, home_team: str, away_team: str, neutral: bool = False
    ) -> float:
        """Get model's estimated home win probability."""
        home = self._get_or_create(home_team)
        away = self._get_or_create(away_team)

        home_adj = home.rating + (0 if neutral else self.config.home_advantage)
        return self.expected_score(home_adj, away.rating)

    def _mov_multiplier(self, margin: float, elo_diff: float) -> float:
        """Margin of victory multiplier to scale K factor.

        Uses log-based formula to prevent extreme updates from blowouts.
        Formula: ln(abs(margin) + 1) * (2.2 / (elo_diff * 0.001 + 2.2))
        """
        if self.config.mov_multiplier == 0:
            return 1.0

        abs_margin = abs(margin)
        log_margin = math.log(abs_margin + 1)

        # Autocorrelation correction: reduce update when strong team wins big
        autocorr = 2.2 / (elo_diff * 0.001 + 2.2)

        return log_margin * autocorr * self.config.mov_multiplier

    def update(
        self,
        home_team: str,
        away_team: str,
        home_score: float,
        away_score: float,
        neutral: bool = False,
    ) -> tuple[float, float]:
        """Update ratings after a game. Returns (home_change, away_change)."""
        home = self._get_or_create(home_team)
        away = self._get_or_create(away_team)

        # Pre-game ratings with home advantage
        home_adj = home.rating + (0 if neutral else self.config.home_advantage)
        expected_home = self.expected_score(home_adj, away.rating)

        # Actual result
        if home_score > away_score:
            actual_home = 1.0
            home.wins += 1
            away.losses += 1
        elif home_score < away_score:
            actual_home = 0.0
            home.losses += 1
            away.wins += 1
        else:
            actual_home = 0.5

        # Margin of victory adjustment
        margin = home_score - away_score
        elo_diff = abs(home_adj - away.rating)
        mov_mult = self._mov_multiplier(margin, elo_diff)

        # Elo update
        k = self.config.k_factor * mov_mult
        change = k * (actual_home - expected_home)

        home.rating += change
        away.rating -= change
        home.games_played += 1
        away.games_played += 1

        # Record history
        self.history.append({
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "home_elo_pre": home_adj,
            "away_elo_pre": away.rating + change,  # pre-update
            "expected_home": expected_home,
            "home_elo_change": change,
            "home_elo_post": home.rating,
            "away_elo_post": away.rating,
        })

        return change, -change

    def season_reset(self) -> None:
        """Revert all ratings toward mean for new season."""
        reversion = self.config.season_reversion
        mean = self.config.mean_rating

        for team_rating in self.ratings.values():
            team_rating.rating = (
                team_rating.rating * (1 - reversion) + mean * reversion
            )
            team_rating.games_played = 0
            team_rating.wins = 0
            team_rating.losses = 0

    def process_season(self, games: pd.DataFrame) -> pd.DataFrame:
        """Process an entire season of games through the Elo model.

        Expected columns: home_team, away_team, home_score, away_score
        Optional: neutral_site (bool)

        Returns DataFrame with pre/post-game Elo ratings and predictions.
        """
        results = []
        for _, game in games.iterrows():
            home = game["home_team"]
            away = game["away_team"]
            neutral = game.get("neutral_site", False)

            # Pre-game prediction
            wp = self.win_probability(home, away, neutral=neutral)

            # Update
            self.update(
                home, away,
                game["home_score"], game["away_score"],
                neutral=neutral,
            )

            results.append({
                "home_team": home,
                "away_team": away,
                "home_score": game["home_score"],
                "away_score": game["away_score"],
                "home_win_prob": wp,
                "home_win": 1 if game["home_score"] > game["away_score"] else 0,
                "home_elo": self.ratings[home].rating,
                "away_elo": self.ratings[away].rating,
            })

        return pd.DataFrame(results)

    def get_rankings(self) -> pd.DataFrame:
        """Get current Elo rankings for all teams."""
        records = [
            {
                "team": r.team,
                "elo": r.rating,
                "games": r.games_played,
                "wins": r.wins,
                "losses": r.losses,
                "win_pct": r.wins / r.games_played if r.games_played > 0 else 0.0,
            }
            for r in self.ratings.values()
        ]
        return (
            pd.DataFrame(records)
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )

    def predict_spread(self, home_team: str, away_team: str, neutral: bool = False) -> float:
        """Predict point spread (positive = home favored).

        Uses the 25-point-per-Elo-point conversion factor.
        """
        home = self._get_or_create(home_team)
        away = self._get_or_create(away_team)

        diff = home.rating - away.rating
        if not neutral:
            diff += self.config.home_advantage

        # ~25 Elo points per point of spread
        return diff / 25.0


def create_nfl_elo(k_factor: float = 20.0, home_adv: float = 48.0) -> EloModel:
    """Create an Elo model tuned for NFL."""
    return EloModel(EloConfig(
        k_factor=k_factor,
        home_advantage=home_adv,
        mean_rating=1500.0,
        season_reversion=0.33,
        mov_multiplier=1.0,
    ))


def create_nba_elo(k_factor: float = 20.0, home_adv: float = 100.0) -> EloModel:
    """Create an Elo model tuned for NBA."""
    return EloModel(EloConfig(
        k_factor=k_factor,
        home_advantage=home_adv,
        mean_rating=1500.0,
        season_reversion=0.25,
        mov_multiplier=0.8,
    ))


def create_mlb_elo(k_factor: float = 4.0, home_adv: float = 24.0) -> EloModel:
    """Create an Elo model tuned for MLB."""
    return EloModel(EloConfig(
        k_factor=k_factor,
        home_advantage=home_adv,
        mean_rating=1500.0,
        season_reversion=0.40,
        mov_multiplier=0.6,
    ))

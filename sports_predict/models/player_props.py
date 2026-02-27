"""Hierarchical Bayesian model for player prop distributions.

Models player stat distributions with:
- Population-level priors from position/role
- Individual-level estimates that shrink toward population
- Opponent/matchup adjustments
- Recent form weighting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class PropPrediction:
    """Prediction for a player prop market."""

    player: str
    stat: str
    line: float
    over_prob: float
    under_prob: float
    projected_value: float
    std_dev: float
    distribution: str  # 'normal', 'poisson', 'negative_binomial'
    percentiles: dict  # {10: val, 25: val, 50: val, 75: val, 90: val}


class PlayerPropModel:
    """Bayesian-inspired player prop model using empirical Bayes.

    Uses conjugate prior updates for computational efficiency instead of
    full MCMC sampling. Falls back to scipy distributions.
    """

    def __init__(self, sport: str = "nfl"):
        self.sport = sport
        self._population_params: dict[str, dict] = {}
        self._player_params: dict[str, dict] = {}

    def fit_population(self, player_stats: pd.DataFrame, stat_col: str) -> dict:
        """Estimate population-level parameters for a stat.

        Args:
            player_stats: DataFrame with player game logs
            stat_col: Column name for the statistic

        Returns:
            Population parameters (mean, var, shape parameters)
        """
        values = player_stats[stat_col].dropna()
        if len(values) < 10:
            return {"mu": values.mean() if len(values) > 0 else 0, "sigma": 1.0, "n": len(values)}

        mu = values.mean()
        sigma = values.std()
        variance = values.var()

        # Determine best distribution
        if all(values == values.astype(int)) and mu > 0:
            # Count data → try negative binomial
            if variance > mu:
                # Overdispersed → negative binomial
                r = mu ** 2 / (variance - mu) if variance > mu else mu
                p = mu / variance if variance > 0 else 0.5
                params = {
                    "distribution": "negative_binomial",
                    "mu": mu, "sigma": sigma,
                    "r": max(r, 0.1), "p": np.clip(p, 0.01, 0.99),
                    "n": len(values),
                }
            else:
                # Poisson
                params = {
                    "distribution": "poisson",
                    "mu": mu, "sigma": sigma,
                    "lambda": mu,
                    "n": len(values),
                }
        else:
            # Continuous → normal
            params = {
                "distribution": "normal",
                "mu": mu, "sigma": max(sigma, 0.1),
                "n": len(values),
            }

        self._population_params[stat_col] = params
        return params

    def fit_player(
        self,
        player_name: str,
        game_logs: pd.DataFrame,
        stat_col: str,
        recency_weight: float = 0.7,
    ) -> dict:
        """Estimate player-specific parameters with shrinkage toward population.

        Uses empirical Bayes: player estimate is weighted average of
        player data and population prior, with weight depending on sample size.
        """
        values = game_logs[stat_col].dropna()
        pop = self._population_params.get(stat_col)

        if pop is None:
            # No population prior, use player data directly
            mu = values.mean() if len(values) > 0 else 0
            sigma = values.std() if len(values) > 1 else 1.0
        else:
            # Empirical Bayes shrinkage
            n = len(values)
            pop_mu = pop["mu"]
            pop_sigma = pop["sigma"]

            if n == 0:
                mu = pop_mu
                sigma = pop_sigma
            else:
                player_mu = values.mean()
                player_sigma = values.std() if n > 1 else pop_sigma

                # Shrinkage factor: more data → less shrinkage
                # B = sigma_pop^2 / (sigma_pop^2 + sigma_player^2/n)
                shrinkage = pop_sigma ** 2 / (
                    pop_sigma ** 2 + player_sigma ** 2 / max(n, 1)
                )
                mu = pop_mu + shrinkage * (player_mu - pop_mu)
                sigma = np.sqrt(
                    (1 - shrinkage) * pop_sigma ** 2 + shrinkage * player_sigma ** 2
                )

        # Apply recency weighting
        if len(values) >= 5:
            recent = values.tail(5)
            recent_mu = recent.mean()
            mu = recency_weight * recent_mu + (1 - recency_weight) * mu

        params = {
            "mu": mu,
            "sigma": max(sigma, 0.1),
            "n_games": len(values),
            "distribution": pop.get("distribution", "normal") if pop else "normal",
        }

        key = f"{player_name}_{stat_col}"
        self._player_params[key] = params
        return params

    def predict_prop(
        self,
        player_name: str,
        stat_col: str,
        line: float,
        opponent_adjustment: float = 0.0,
    ) -> PropPrediction:
        """Predict over/under probability for a player prop.

        Args:
            player_name: Player name
            stat_col: Stat being predicted
            line: The prop line (e.g., 275.5 passing yards)
            opponent_adjustment: Multiplicative adjustment for opponent strength
                                (1.1 = opponent allows 10% more than average)
        """
        key = f"{player_name}_{stat_col}"
        params = self._player_params.get(key)

        if params is None:
            # No data, use population
            params = self._population_params.get(stat_col, {
                "mu": line, "sigma": line * 0.3, "distribution": "normal"
            })

        mu = params["mu"] * (1 + opponent_adjustment)
        sigma = params["sigma"]
        dist_type = params["distribution"]

        # Calculate probabilities
        if dist_type == "poisson":
            lam = max(mu, 0.1)
            over_prob = 1 - sp_stats.poisson.cdf(int(line), lam)
            under_prob = sp_stats.poisson.cdf(int(line), lam)
            percentiles = {
                p: float(sp_stats.poisson.ppf(p / 100, lam))
                for p in [10, 25, 50, 75, 90]
            }
        elif dist_type == "negative_binomial":
            pop = self._population_params.get(stat_col, {})
            r = pop.get("r", mu)
            p_param = r / (r + mu) if (r + mu) > 0 else 0.5
            over_prob = 1 - sp_stats.nbinom.cdf(int(line), r, p_param)
            under_prob = sp_stats.nbinom.cdf(int(line), r, p_param)
            percentiles = {
                p: float(sp_stats.nbinom.ppf(p / 100, r, p_param))
                for p in [10, 25, 50, 75, 90]
            }
        else:
            # Normal distribution
            over_prob = 1 - sp_stats.norm.cdf(line, mu, sigma)
            under_prob = sp_stats.norm.cdf(line, mu, sigma)
            percentiles = {
                p: float(sp_stats.norm.ppf(p / 100, mu, sigma))
                for p in [10, 25, 50, 75, 90]
            }

        return PropPrediction(
            player=player_name,
            stat=stat_col,
            line=line,
            over_prob=round(over_prob, 4),
            under_prob=round(under_prob, 4),
            projected_value=round(mu, 1),
            std_dev=round(sigma, 1),
            distribution=dist_type,
            percentiles=percentiles,
        )

    def batch_predict(
        self,
        player_game_logs: pd.DataFrame,
        stat_col: str,
        props: list[tuple[str, float]],
        opponent_adjustments: dict[str, float] | None = None,
    ) -> list[PropPrediction]:
        """Predict multiple player props in batch.

        Args:
            player_game_logs: DataFrame with player game logs
            stat_col: Stat column to predict
            props: List of (player_name, line) tuples
            opponent_adjustments: Dict of player -> adjustment factor

        Returns:
            List of PropPrediction objects
        """
        # Fit population first
        self.fit_population(player_game_logs, stat_col)

        # Fit each player
        player_col = "player_name" if "player_name" in player_game_logs.columns else "PLAYER_NAME"
        for player_name, _ in props:
            player_data = player_game_logs[
                player_game_logs[player_col] == player_name
            ]
            if not player_data.empty:
                self.fit_player(player_name, player_data, stat_col)

        # Generate predictions
        results = []
        for player_name, line in props:
            adj = (opponent_adjustments or {}).get(player_name, 0.0)
            pred = self.predict_prop(player_name, stat_col, line, adj)
            results.append(pred)

        return results

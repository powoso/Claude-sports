"""Regularized adjusted efficiency model for game predictions.

Combines EPA/net-rating efficiency metrics with situational factors
to produce calibrated win probabilities and projected scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class GamePrediction:
    """Prediction output for a single game."""

    home_team: str
    away_team: str
    home_win_prob: float
    away_win_prob: float
    projected_home_score: float
    projected_away_score: float
    projected_total: float
    projected_spread: float  # positive = home favored
    confidence: float  # 0-1, model confidence
    components: dict | None = None  # breakdown of prediction factors


class EfficiencyModel:
    """Regularized efficiency model that combines multiple signals.

    Uses ridge regression on efficiency differentials + situational factors
    to predict outcomes, then calibrates probabilities using isotonic regression.
    """

    def __init__(self, sport: str = "nfl"):
        self.sport = sport
        self.scaler = StandardScaler()
        self.model = RidgeClassifier(alpha=1.0)
        self.calibrated_model = None
        self.is_fitted = False
        self._feature_names: list[str] = []

        # Sport-specific defaults
        self._defaults = {
            "nfl": {"avg_total": 45.0, "home_edge": 2.5, "pts_per_elo": 25.0},
            "nba": {"avg_total": 225.0, "home_edge": 3.5, "pts_per_elo": 28.0},
            "mlb": {"avg_total": 8.5, "home_edge": 0.3, "pts_per_elo": 50.0},
        }.get(sport, {"avg_total": 45.0, "home_edge": 2.5, "pts_per_elo": 25.0})

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Select numeric feature columns for model input."""
        exclude = {
            "game_id", "season", "week", "home_team", "away_team",
            "home_score", "away_score", "home_win", "margin",
            "game_date", "spread_line", "total_line",
        }
        return [
            c for c in df.columns
            if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]
        ]

    def fit(self, features: pd.DataFrame, target_col: str = "home_win") -> None:
        """Train the efficiency model on historical game features.

        Args:
            features: DataFrame with game-level features and outcome
            target_col: Column name for binary outcome (1=home win)
        """
        # Drop rows without outcomes
        df = features.dropna(subset=[target_col]).copy()
        if len(df) < 20:
            logger.warning("Too few games (%d) to train model", len(df))
            return

        self._feature_names = self._get_feature_cols(df)
        X = df[self._feature_names].fillna(0).values
        y = df[target_col].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit calibrated model
        self.calibrated_model = CalibratedClassifierCV(
            self.model, cv=min(5, len(df) // 10), method="isotonic"
        )
        self.calibrated_model.fit(X_scaled, y)
        self.is_fitted = True

        # Log feature importances (from base ridge model)
        base = self.calibrated_model.estimator
        if hasattr(base, "coef_"):
            importances = pd.Series(
                base.coef_[0] if base.coef_.ndim > 1 else base.coef_,
                index=self._feature_names,
            ).abs().sort_values(ascending=False)
            logger.info("Top features:\n%s", importances.head(10))

    def predict_game(
        self,
        features: dict | pd.Series,
        elo_home_wp: float | None = None,
    ) -> GamePrediction:
        """Predict a single game outcome.

        Args:
            features: Game feature dictionary or Series
            elo_home_wp: Optional Elo-based win probability to blend

        Returns:
            GamePrediction with calibrated probabilities and projected scores
        """
        home_team = features.get("home_team", "HOME")
        away_team = features.get("away_team", "AWAY")

        if self.is_fitted and self._feature_names:
            # Use trained model
            X = np.array([[features.get(f, 0) for f in self._feature_names]])
            X_scaled = self.scaler.transform(X)
            probs = self.calibrated_model.predict_proba(X_scaled)[0]
            efficiency_wp = probs[1]  # probability of class 1 (home win)
        else:
            # Fallback: use feature differentials directly
            efficiency_wp = self._heuristic_prediction(features)

        # Blend with Elo if available
        if elo_home_wp is not None:
            # Weighted blend: 60% efficiency model, 40% Elo
            home_wp = 0.6 * efficiency_wp + 0.4 * elo_home_wp
        else:
            home_wp = efficiency_wp

        # Clamp probabilities
        home_wp = np.clip(home_wp, 0.01, 0.99)

        # Project scores from win probability and total
        projected_total = features.get("total_line", self._defaults["avg_total"])
        spread = self._wp_to_spread(home_wp)

        projected_home = (projected_total / 2) + (spread / 2)
        projected_away = (projected_total / 2) - (spread / 2)

        # Confidence based on probability distance from 0.5
        confidence = abs(home_wp - 0.5) * 2  # 0 at 50/50, 1 at certainty

        return GamePrediction(
            home_team=home_team,
            away_team=away_team,
            home_win_prob=home_wp,
            away_win_prob=1.0 - home_wp,
            projected_home_score=round(projected_home, 1),
            projected_away_score=round(projected_away, 1),
            projected_total=round(projected_total, 1),
            projected_spread=round(spread, 1),
            confidence=round(confidence, 3),
            components={
                "efficiency_wp": round(efficiency_wp, 4),
                "elo_wp": round(elo_home_wp, 4) if elo_home_wp else None,
                "blended_wp": round(home_wp, 4),
            },
        )

    def _heuristic_prediction(self, features: dict) -> float:
        """Fallback prediction using simple feature differentials."""
        # Look for common differential features
        diff_signals = []

        for key in ["diff_epa_play_rolling", "net_rtg_diff", "wOBA_diff"]:
            val = features.get(key)
            if val is not None and not np.isnan(val):
                diff_signals.append(val)

        if not diff_signals:
            return 0.5 + self._defaults["home_edge"] / 100.0

        avg_diff = np.mean(diff_signals)
        # Convert to probability using logistic function
        # Scale factor depends on sport
        scale = {"nfl": 5.0, "nba": 10.0, "mlb": 3.0}.get(self.sport, 5.0)
        prob = 1.0 / (1.0 + np.exp(-avg_diff * scale))

        return prob

    @staticmethod
    def _wp_to_spread(win_prob: float) -> float:
        """Convert win probability to projected point spread.

        Uses inverse normal CDF mapping.
        """
        if win_prob <= 0.01 or win_prob >= 0.99:
            return np.clip(stats.norm.ppf(win_prob) * 7, -28, 28)
        return stats.norm.ppf(win_prob) * 7  # ~7 points per std dev

    def evaluate(self, predictions: pd.DataFrame) -> dict:
        """Evaluate model predictions against actuals.

        Args:
            predictions: DataFrame with 'home_win_prob' and 'home_win' columns

        Returns:
            Dictionary of evaluation metrics
        """
        df = predictions.dropna(subset=["home_win_prob", "home_win"])
        if df.empty:
            return {}

        probs = df["home_win_prob"].values
        actuals = df["home_win"].values

        # Accuracy
        picks = (probs >= 0.5).astype(int)
        accuracy = (picks == actuals).mean()

        # Brier score (lower is better)
        brier = np.mean((probs - actuals) ** 2)

        # Log loss
        eps = 1e-10
        clipped = np.clip(probs, eps, 1 - eps)
        log_loss = -np.mean(
            actuals * np.log(clipped) + (1 - actuals) * np.log(1 - clipped)
        )

        # Calibration: bin predictions and compare to actual win rate
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        calibration = []
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if mask.sum() > 0:
                calibration.append({
                    "bin_center": (bin_edges[i] + bin_edges[i + 1]) / 2,
                    "predicted": probs[mask].mean(),
                    "actual": actuals[mask].mean(),
                    "count": int(mask.sum()),
                })

        return {
            "accuracy": round(accuracy, 4),
            "brier_score": round(brier, 4),
            "log_loss": round(log_loss, 4),
            "n_games": len(df),
            "calibration": calibration,
        }

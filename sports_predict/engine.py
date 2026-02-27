"""Main orchestration engine that ties together data, features, models, and edge detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from sports_predict.data_collection.nfl import NFLDataCollector
from sports_predict.data_collection.nba import NBADataCollector
from sports_predict.data_collection.mlb import MLBDataCollector
from sports_predict.data_collection.odds import OddsCollector, Sport
from sports_predict.data_collection.weather import WeatherCollector
from sports_predict.edge_detection.edge_finder import EdgeFinder, LineMovementTracker
from sports_predict.features.mlb_features import MLBFeatureEngine
from sports_predict.features.nba_features import NBAFeatureEngine
from sports_predict.features.nfl_features import NFLFeatureEngine
from sports_predict.market.portfolio import PortfolioManager
from sports_predict.models.elo import EloModel, create_mlb_elo, create_nba_elo, create_nfl_elo
from sports_predict.models.efficiency import EfficiencyModel
from sports_predict.models.player_props import PlayerPropModel
from sports_predict.models.season_sim import SeasonSimulator
from sports_predict.models.totals import TotalsModel
from sports_predict.utils.cache import DataCache
from sports_predict.utils.config import Settings, load_settings

logger = logging.getLogger(__name__)


@dataclass
class SportEngine:
    """Holds all components for a single sport."""

    collector: object
    feature_engine: object
    elo: EloModel
    efficiency: EfficiencyModel
    props: PlayerPropModel
    totals: TotalsModel


class PredictionEngine:
    """Main orchestration engine for the sports prediction system.

    Coordinates data collection, feature engineering, model predictions,
    and edge detection across all supported sports.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.cache = DataCache(
            cache_dir=self.settings.data.cache_dir,
            ttl_hours=self.settings.data.cache_ttl_hours,
        )

        # Data collectors
        self.odds = OddsCollector(cache=self.cache)
        self.weather = WeatherCollector(cache=self.cache)

        # Sport-specific engines
        self.nfl = self._build_nfl()
        self.nba = self._build_nba()
        self.mlb = self._build_mlb()

        # Cross-sport components
        self.edge_finder = EdgeFinder(
            min_edge_pct=self.settings.edge.min_edge_pct,
            kelly_fraction=self.settings.edge.kelly_fraction,
            max_position_pct=self.settings.edge.max_position_pct,
        )
        self.line_tracker = LineMovementTracker(
            sharp_move_threshold=self.settings.edge.sharp_move_threshold,
        )
        self.season_sim = SeasonSimulator(
            n_simulations=self.settings.models.monte_carlo_iterations,
        )
        self.portfolio = PortfolioManager()

    def _build_nfl(self) -> SportEngine:
        cfg = self.settings.nfl
        collector = NFLDataCollector(seasons=cfg.seasons, cache=self.cache)
        features = NFLFeatureEngine(
            collector=collector,
            weather=self.weather,
            smoothing_window=cfg.epa_smoothing_window,
        )
        return SportEngine(
            collector=collector,
            feature_engine=features,
            elo=create_nfl_elo(k_factor=cfg.elo_k_factor, home_adv=cfg.elo_home_advantage),
            efficiency=EfficiencyModel(sport="nfl"),
            props=PlayerPropModel(sport="nfl"),
            totals=TotalsModel(sport="nfl"),
        )

    def _build_nba(self) -> SportEngine:
        cfg = self.settings.nba
        collector = NBADataCollector(seasons=cfg.seasons, cache=self.cache)
        features = NBAFeatureEngine(
            collector=collector,
            rolling_window=cfg.net_rating_window,
            rest_penalty_b2b=cfg.rest_penalty_b2b,
            travel_penalty_per_1000mi=cfg.travel_penalty_per_1000mi,
        )
        return SportEngine(
            collector=collector,
            feature_engine=features,
            elo=create_nba_elo(),
            efficiency=EfficiencyModel(sport="nba"),
            props=PlayerPropModel(sport="nba"),
            totals=TotalsModel(sport="nba"),
        )

    def _build_mlb(self) -> SportEngine:
        cfg = self.settings.mlb
        collector = MLBDataCollector(seasons=cfg.seasons, cache=self.cache)
        features = MLBFeatureEngine(
            collector=collector,
            xwoba_smoothing_pa=cfg.xwoba_smoothing_pa,
        )
        return SportEngine(
            collector=collector,
            feature_engine=features,
            elo=create_mlb_elo(),
            efficiency=EfficiencyModel(sport="mlb"),
            props=PlayerPropModel(sport="mlb"),
            totals=TotalsModel(sport="mlb"),
        )

    def run_nfl_predictions(self, season: int, week: int) -> dict:
        """Run full NFL prediction pipeline for a given week.

        Returns dict with predictions, edges, and model diagnostics.
        """
        logger.info("Running NFL predictions: season=%d, week=%d", season, week)

        # Build features
        features = self.nfl.feature_engine.build_game_features(season, week)
        if features.empty:
            logger.warning("No games found for NFL season %d week %d", season, week)
            return {"games": [], "edges": []}

        # Get Elo predictions
        predictions = []
        for _, game in features.iterrows():
            home = game["home_team"]
            away = game["away_team"]

            elo_wp = self.nfl.elo.win_probability(home, away)
            elo_spread = self.nfl.elo.predict_spread(home, away)

            # Efficiency model prediction
            eff_pred = self.nfl.efficiency.predict_game(game.to_dict(), elo_home_wp=elo_wp)

            predictions.append({
                "game_id": game.get("game_id", f"{home}_{away}"),
                "home_team": home,
                "away_team": away,
                "home_win_prob": eff_pred.home_win_prob,
                "projected_spread": eff_pred.projected_spread,
                "projected_total": eff_pred.projected_total,
                "projected_home_score": eff_pred.projected_home_score,
                "projected_away_score": eff_pred.projected_away_score,
                "elo_wp": round(elo_wp, 4),
                "elo_spread": round(elo_spread, 1),
                "confidence": eff_pred.confidence,
            })

        # Fetch market prices and find edges
        market_prices = self.odds.get_all_market_prices(Sport.NFL)
        model_probs = {p["game_id"]: p["home_win_prob"] for p in predictions}
        edges = self.edge_finder.find_game_edges(model_probs, market_prices, "nfl")

        return {
            "season": season,
            "week": week,
            "games": predictions,
            "edges": [
                {
                    "event": e.event_name,
                    "outcome": e.outcome,
                    "model_prob": e.model_prob,
                    "market_prob": e.market_prob,
                    "edge_pct": e.edge_pct,
                    "kelly_size": e.recommended_size_pct,
                    "source": e.market_source,
                    "biases": [b.value for b in e.detected_biases],
                }
                for e in edges
            ],
        }

    def run_nba_predictions(self, season: str, game_date: str | None = None) -> dict:
        """Run full NBA prediction pipeline."""
        logger.info("Running NBA predictions: season=%s, date=%s", season, game_date)

        features = self.nba.feature_engine.build_game_features(season, game_date)
        if features.empty:
            return {"games": [], "edges": []}

        predictions = []
        for _, game in features.iterrows():
            home = game["team"]
            away = game.get("away_team", "")

            elo_wp = self.nba.elo.win_probability(home, away)
            eff_pred = self.nba.efficiency.predict_game(game.to_dict(), elo_home_wp=elo_wp)

            predictions.append({
                "game_id": game.get("game_id", f"{home}_{away}"),
                "home_team": home,
                "away_team": away,
                "home_win_prob": eff_pred.home_win_prob,
                "projected_spread": eff_pred.projected_spread,
                "projected_total": eff_pred.projected_total,
                "elo_wp": round(elo_wp, 4),
                "confidence": eff_pred.confidence,
            })

        market_prices = self.odds.get_all_market_prices(Sport.NBA)
        model_probs = {p["game_id"]: p["home_win_prob"] for p in predictions}
        edges = self.edge_finder.find_game_edges(model_probs, market_prices, "nba")

        return {
            "season": season,
            "game_date": game_date,
            "games": predictions,
            "edges": [
                {
                    "event": e.event_name,
                    "outcome": e.outcome,
                    "model_prob": e.model_prob,
                    "market_prob": e.market_prob,
                    "edge_pct": e.edge_pct,
                    "kelly_size": e.recommended_size_pct,
                    "source": e.market_source,
                }
                for e in edges
            ],
        }

    def run_season_simulation(self, sport: str) -> dict:
        """Run Monte Carlo season simulation for a sport.

        Returns win totals, playoff probabilities, and season-long edges.
        """
        logger.info("Running season simulation for %s", sport)

        if sport == "nfl":
            engine = self.nfl
            latest_season = self.settings.nfl.seasons[-1]
            schedule = engine.collector.get_schedules(latest_season)

            # Need home_team, away_team columns
            if "home_team" not in schedule.columns:
                return {"error": "Schedule format not recognized"}

            result = self.season_sim.simulate_nfl_season(
                schedule=schedule,
                get_win_prob=lambda h, a: engine.elo.win_probability(h, a),
            )
        elif sport == "nba":
            # Would need NBA schedule in similar format
            return {"error": "NBA season sim requires schedule data"}
        else:
            return {"error": f"Season sim not implemented for {sport}"}

        outcomes = []
        for o in result.team_outcomes:
            outcomes.append({
                "team": o.team,
                "mean_wins": round(o.mean_wins, 1),
                "median_wins": round(o.median_wins, 0),
                "std_wins": round(o.std_wins, 1),
                "playoff_prob": round(o.playoff_prob, 3),
                "division_winner_prob": round(o.division_winner_prob, 3),
            })

        return {
            "sport": sport,
            "n_simulations": result.n_simulations,
            "team_outcomes": outcomes,
        }

    def get_portfolio_summary(self) -> dict:
        """Get current portfolio status."""
        return self.portfolio.get_summary()

    def close(self) -> None:
        """Clean up resources."""
        self.odds.close()
        self.weather.close()

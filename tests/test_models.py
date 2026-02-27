"""Tests for efficiency model, player props, season sim, and totals."""

import numpy as np
import pandas as pd
import pytest

from sports_predict.models.efficiency import EfficiencyModel, GamePrediction
from sports_predict.models.player_props import PlayerPropModel, PropPrediction
from sports_predict.models.season_sim import SeasonSimulator
from sports_predict.models.totals import TotalsModel


class TestEfficiencyModel:
    def test_heuristic_prediction_home_bias(self):
        model = EfficiencyModel(sport="nfl")
        # No differential features → slight home advantage
        pred = model.predict_game({"home_team": "A", "away_team": "B"})
        assert pred.home_win_prob > 0.5

    def test_heuristic_with_differential(self):
        model = EfficiencyModel(sport="nfl")
        pred = model.predict_game({
            "home_team": "A",
            "away_team": "B",
            "diff_epa_play_rolling": 0.15,
        })
        # Positive differential → home team should be more favored
        assert pred.home_win_prob > 0.55

    def test_prediction_with_elo_blend(self):
        model = EfficiencyModel(sport="nfl")
        pred = model.predict_game(
            {"home_team": "A", "away_team": "B", "diff_epa_play_rolling": 0.1},
            elo_home_wp=0.70,
        )
        # Should blend toward Elo
        assert pred.home_win_prob > 0.55
        assert pred.components["elo_wp"] == 0.7

    def test_prediction_has_projected_scores(self):
        model = EfficiencyModel(sport="nfl")
        pred = model.predict_game({
            "home_team": "A", "away_team": "B",
            "total_line": 45.0,
        })
        assert pred.projected_total == 45.0
        assert pred.projected_home_score > 0
        assert pred.projected_away_score > 0

    def test_probability_bounds(self):
        model = EfficiencyModel(sport="nfl")
        # Extreme differential
        pred = model.predict_game({
            "home_team": "A", "away_team": "B",
            "diff_epa_play_rolling": 1.0,  # very large
        })
        assert 0.01 <= pred.home_win_prob <= 0.99

    def test_evaluate_metrics(self):
        model = EfficiencyModel(sport="nfl")
        predictions = pd.DataFrame({
            "home_win_prob": [0.6, 0.7, 0.4, 0.8, 0.55],
            "home_win": [1, 1, 0, 1, 0],
        })
        metrics = model.evaluate(predictions)
        assert "accuracy" in metrics
        assert "brier_score" in metrics
        assert "log_loss" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["brier_score"] >= 0
        assert metrics["n_games"] == 5


class TestPlayerPropModel:
    def setup_method(self):
        self.model = PlayerPropModel(sport="nfl")

    def test_fit_population_normal(self):
        data = pd.DataFrame({"passing_yards": np.random.normal(250, 60, 100)})
        params = self.model.fit_population(data, "passing_yards")
        assert params["distribution"] == "normal"
        assert abs(params["mu"] - 250) < 20

    def test_fit_population_count(self):
        data = pd.DataFrame({"touchdowns": np.random.poisson(2.0, 100).astype(float)})
        params = self.model.fit_population(data, "touchdowns")
        assert params["distribution"] in ("poisson", "negative_binomial")

    def test_fit_player_shrinkage(self):
        # Fit population
        pop_data = pd.DataFrame({"yards": np.random.normal(250, 60, 200)})
        self.model.fit_population(pop_data, "yards")

        # Player with small sample (extreme value should shrink toward pop)
        player_data = pd.DataFrame({"yards": [400, 380, 350]})
        params = self.model.fit_player("TestQB", player_data, "yards")

        # Should be between population mean and player mean due to shrinkage
        assert params["mu"] < 400  # shrank from player mean
        assert params["mu"] > 250  # but above population mean

    def test_predict_prop(self):
        pop_data = pd.DataFrame({"yards": np.random.normal(250, 60, 200)})
        self.model.fit_population(pop_data, "yards")

        player_data = pd.DataFrame({"yards": np.random.normal(280, 50, 16)})
        self.model.fit_player("Mahomes", player_data, "yards")

        pred = self.model.predict_prop("Mahomes", "yards", 275.5)
        assert isinstance(pred, PropPrediction)
        assert 0 < pred.over_prob < 1
        assert 0 < pred.under_prob < 1
        assert pred.over_prob + pred.under_prob == pytest.approx(1.0, abs=0.01)

    def test_predict_prop_over_favored(self):
        pop_data = pd.DataFrame({"yards": np.random.normal(250, 60, 200)})
        self.model.fit_population(pop_data, "yards")

        # Player who consistently gets ~300 yards
        player_data = pd.DataFrame({"yards": [300, 310, 290, 305, 295]})
        self.model.fit_player("Elite", player_data, "yards")

        # Line set way below projection
        pred = self.model.predict_prop("Elite", "yards", 200.0)
        assert pred.over_prob > 0.7

    def test_opponent_adjustment(self):
        pop_data = pd.DataFrame({"yards": np.random.normal(250, 60, 200)})
        self.model.fit_population(pop_data, "yards")

        player_data = pd.DataFrame({"yards": [280, 270, 290, 285, 275]})
        self.model.fit_player("QB", player_data, "yards")

        pred_neutral = self.model.predict_prop("QB", "yards", 280)
        pred_weak_def = self.model.predict_prop("QB", "yards", 280, opponent_adjustment=0.10)

        # Against weak defense, over should be more likely
        assert pred_weak_def.over_prob > pred_neutral.over_prob

    def test_batch_predict(self):
        game_logs = pd.DataFrame({
            "player_name": ["A"] * 10 + ["B"] * 10,
            "yards": list(np.random.normal(250, 50, 10)) + list(np.random.normal(200, 40, 10)),
        })

        props = [("A", 260.0), ("B", 210.0)]
        results = self.model.batch_predict(game_logs, "yards", props)
        assert len(results) == 2


class TestSeasonSimulator:
    def test_nfl_season_sim(self):
        sim = SeasonSimulator(n_simulations=100, seed=42)

        # Create a simple 4-team schedule
        games = []
        teams = ["KC", "BUF", "BAL", "DET"]
        for i, home in enumerate(teams):
            for j, away in enumerate(teams):
                if i != j:
                    games.append({"home_team": home, "away_team": away})

        schedule = pd.DataFrame(games)

        result = sim.simulate_nfl_season(
            schedule,
            get_win_prob=lambda h, a: 0.6,  # home always has 60%
        )

        assert result.n_simulations == 100
        assert len(result.team_outcomes) == 4

        # All teams should have some wins
        for o in result.team_outcomes:
            assert o.mean_wins > 0
            assert o.std_wins > 0

    def test_win_total_probs(self):
        sim = SeasonSimulator(n_simulations=1000, seed=42)

        schedule = pd.DataFrame([
            {"home_team": "A", "away_team": "B"},
            {"home_team": "B", "away_team": "A"},
            {"home_team": "A", "away_team": "B"},
            {"home_team": "B", "away_team": "A"},
        ])

        result = sim.simulate_nfl_season(
            schedule,
            get_win_prob=lambda h, a: 0.7,  # strong home advantage
        )

        # Find team A's outcome
        a_outcome = next(o for o in result.team_outcomes if o.team == "A")

        over, under = sim.compute_win_total_probs(a_outcome, 2.0)
        assert 0 <= over <= 1
        assert 0 <= under <= 1
        assert over + under == pytest.approx(1.0, abs=0.1)

    def test_simulation_reproducible(self):
        schedule = pd.DataFrame([
            {"home_team": "A", "away_team": "B"},
            {"home_team": "B", "away_team": "A"},
        ])

        sim1 = SeasonSimulator(n_simulations=100, seed=42)
        result1 = sim1.simulate_nfl_season(schedule, get_win_prob=lambda h, a: 0.6)

        sim2 = SeasonSimulator(n_simulations=100, seed=42)
        result2 = sim2.simulate_nfl_season(schedule, get_win_prob=lambda h, a: 0.6)

        # Same seed should give same results
        for o1, o2 in zip(result1.team_outcomes, result2.team_outcomes):
            assert o1.mean_wins == o2.mean_wins


class TestTotalsModel:
    def test_nfl_total(self):
        model = TotalsModel(sport="nfl")
        pred = model.predict_nfl_total(
            home_off_epa=0.10,
            home_def_epa=-0.05,
            away_off_epa=0.05,
            away_def_epa=0.0,
        )
        assert pred.projected_total > 0
        assert pred.projected_home_score > 0
        assert pred.projected_away_score > 0

    def test_nfl_total_weather_reduction(self):
        model = TotalsModel(sport="nfl")
        normal = model.predict_nfl_total(0.05, 0.0, 0.05, 0.0, weather_factor=1.0)
        bad_weather = model.predict_nfl_total(0.05, 0.0, 0.05, 0.0, weather_factor=0.85)
        assert bad_weather.projected_total < normal.projected_total

    def test_nfl_total_dome_ignores_weather(self):
        model = TotalsModel(sport="nfl")
        dome = model.predict_nfl_total(0.05, 0.0, 0.05, 0.0, weather_factor=0.5, dome=True)
        outdoor = model.predict_nfl_total(0.05, 0.0, 0.05, 0.0, weather_factor=0.5, dome=False)
        assert dome.projected_total > outdoor.projected_total

    def test_nba_total(self):
        model = TotalsModel(sport="nba")
        pred = model.predict_nba_total(
            home_off_rtg=115.0,
            home_def_rtg=108.0,
            away_off_rtg=110.0,
            away_def_rtg=112.0,
            home_pace=100.0,
            away_pace=98.0,
        )
        assert 180 < pred.projected_total < 280

    def test_mlb_total(self):
        model = TotalsModel(sport="mlb")
        pred = model.predict_mlb_total(
            home_woba=0.330,
            away_woba=0.310,
            home_era=3.50,
            away_era=4.20,
            park_factor=1.0,
        )
        assert 4.0 < pred.projected_total < 15.0

    def test_mlb_coors_field_boost(self):
        model = TotalsModel(sport="mlb")
        normal = model.predict_mlb_total(0.320, 0.320, 4.0, 4.0, park_factor=1.0)
        coors = model.predict_mlb_total(0.320, 0.320, 4.0, 4.0, park_factor=1.30)
        assert coors.projected_total > normal.projected_total

    def test_set_line_and_compute(self):
        model = TotalsModel(sport="nfl")
        pred = model.predict_nfl_total(0.05, 0.0, 0.05, 0.0)
        pred = model.set_line_and_compute(pred, 44.5)
        assert pred.line == 44.5
        assert 0 < pred.over_prob < 1
        assert 0 < pred.under_prob < 1
        assert pred.over_prob + pred.under_prob == pytest.approx(1.0, abs=0.01)

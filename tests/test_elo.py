"""Tests for the Elo rating model."""

import numpy as np
import pandas as pd
import pytest

from sports_predict.models.elo import (
    EloConfig,
    EloModel,
    create_mlb_elo,
    create_nba_elo,
    create_nfl_elo,
)


class TestEloModel:
    def setup_method(self):
        self.model = EloModel(EloConfig(
            k_factor=20.0,
            home_advantage=50.0,
            mean_rating=1500.0,
        ))

    def test_initial_rating(self):
        wp = self.model.win_probability("TeamA", "TeamB")
        # With home advantage, home team should be favored
        assert wp > 0.5
        # But not overwhelmingly
        assert wp < 0.7

    def test_expected_score_symmetric(self):
        """Equal ratings should give 50/50 odds."""
        assert self.model.expected_score(1500, 1500) == pytest.approx(0.5)

    def test_expected_score_higher_wins(self):
        """Higher-rated team should be favored."""
        assert self.model.expected_score(1600, 1400) > 0.5
        assert self.model.expected_score(1400, 1600) < 0.5

    def test_expected_score_sums_to_one(self):
        """Probabilities should sum to 1."""
        p = self.model.expected_score(1550, 1450)
        q = self.model.expected_score(1450, 1550)
        assert p + q == pytest.approx(1.0)

    def test_update_winner_gains(self):
        """Winning team should gain Elo points."""
        self.model.update("A", "B", home_score=28, away_score=14)
        assert self.model.ratings["A"].rating > 1500
        assert self.model.ratings["B"].rating < 1500

    def test_update_zero_sum(self):
        """Elo changes should be zero-sum."""
        self.model.update("A", "B", home_score=28, away_score=14)
        total = self.model.ratings["A"].rating + self.model.ratings["B"].rating
        assert total == pytest.approx(3000.0, abs=0.1)

    def test_upset_larger_change(self):
        """An upset (lower-rated team wins) should produce a larger Elo change."""
        # Set up: A is much better
        self.model.ratings = {}
        model1 = EloModel(EloConfig(k_factor=20.0, home_advantage=0))
        model1.update("A", "B", 30, 20)  # Expected result (new teams)
        change1 = model1.history[-1]["home_elo_change"]

        model2 = EloModel(EloConfig(k_factor=20.0, home_advantage=0))
        # Give B a much lower rating
        model2._get_or_create("A").rating = 1600
        model2._get_or_create("B").rating = 1400
        model2.update("B", "A", 30, 20)  # B (underdog) beats A at A's home
        change2 = model2.history[-1]["home_elo_change"]

        # B (home, underdog) winning should create larger positive change
        assert change2 > change1

    def test_season_reset_reverts_to_mean(self):
        """Season reset should move ratings toward the mean."""
        self.model._get_or_create("Strong").rating = 1700
        self.model._get_or_create("Weak").rating = 1300

        self.model.season_reset()

        assert self.model.ratings["Strong"].rating < 1700
        assert self.model.ratings["Weak"].rating > 1300
        # Should still maintain relative ordering
        assert self.model.ratings["Strong"].rating > self.model.ratings["Weak"].rating

    def test_predict_spread(self):
        """Spread prediction should favor higher-rated team."""
        self.model._get_or_create("A").rating = 1600
        self.model._get_or_create("B").rating = 1400

        spread = self.model.predict_spread("A", "B")
        # A is home and higher rated, should be favored (positive spread)
        assert spread > 0

        # Neutral site should reduce the spread
        spread_neutral = self.model.predict_spread("A", "B", neutral=True)
        assert spread_neutral < spread

    def test_process_season(self):
        """Process a full season of games."""
        games = pd.DataFrame({
            "home_team": ["A", "B", "A", "C"],
            "away_team": ["B", "C", "C", "A"],
            "home_score": [28, 21, 35, 17],
            "away_score": [14, 24, 28, 20],
        })

        results = self.model.process_season(games)

        assert len(results) == 4
        assert "home_win_prob" in results.columns
        assert "home_elo" in results.columns
        assert all(0 < p < 1 for p in results["home_win_prob"])

    def test_get_rankings(self):
        """Rankings should be sorted by Elo."""
        games = pd.DataFrame({
            "home_team": ["A", "B", "A"],
            "away_team": ["B", "C", "C"],
            "home_score": [28, 21, 35],
            "away_score": [14, 24, 10],
        })
        self.model.process_season(games)
        rankings = self.model.get_rankings()

        assert len(rankings) == 3
        assert rankings.iloc[0]["elo"] >= rankings.iloc[1]["elo"]

    def test_neutral_site_no_home_advantage(self):
        """Neutral site should eliminate home advantage."""
        wp_home = self.model.win_probability("A", "B", neutral=False)
        wp_neutral = self.model.win_probability("A", "B", neutral=True)

        # Equal teams: home advantage should make home favored
        assert wp_home > wp_neutral
        assert wp_neutral == pytest.approx(0.5)


class TestSportSpecificElo:
    def test_nfl_elo_creation(self):
        model = create_nfl_elo()
        assert model.config.k_factor == 20.0
        assert model.config.season_reversion == 0.33

    def test_nba_elo_creation(self):
        model = create_nba_elo()
        assert model.config.home_advantage == 100.0
        assert model.config.season_reversion == 0.25

    def test_mlb_elo_creation(self):
        model = create_mlb_elo()
        assert model.config.k_factor == 4.0
        assert model.config.season_reversion == 0.40

    def test_nfl_elo_reasonable_predictions(self):
        model = create_nfl_elo()
        wp = model.win_probability("KC", "DEN")
        # Both new teams, home advantage should give ~55-60%
        assert 0.50 < wp < 0.70

    def test_mlb_smaller_updates(self):
        """MLB should have smaller K factor → smaller rating changes."""
        nfl = create_nfl_elo()
        mlb = create_mlb_elo()

        nfl.update("A", "B", 28, 14)
        mlb.update("A", "B", 5, 2)

        nfl_change = abs(nfl.ratings["A"].rating - 1500)
        mlb_change = abs(mlb.ratings["A"].rating - 1500)

        assert mlb_change < nfl_change

"""Monte Carlo season simulation engine.

Simulates full seasons (10,000+ iterations) using game-level win probabilities
to generate distributions for:
- Win totals
- Playoff probabilities
- Division/conference championship probabilities
- Season-long prediction market outcomes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SeasonOutcome:
    """Aggregated outcome from Monte Carlo simulation."""

    team: str
    mean_wins: float
    median_wins: float
    std_wins: float
    win_distribution: dict[int, float]  # {win_count: probability}
    playoff_prob: float
    division_winner_prob: float
    conference_winner_prob: float
    champion_prob: float
    over_under_line: float | None = None
    over_prob: float | None = None


@dataclass
class SimulationResult:
    """Complete simulation output."""

    n_simulations: int
    team_outcomes: list[SeasonOutcome]
    raw_wins: np.ndarray | None = None  # (n_sims, n_teams) matrix


# NFL conference/division structure
NFL_DIVISIONS: dict[str, list[str]] = {
    "AFC_East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC_North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC_South": ["HOU", "IND", "JAX", "TEN"],
    "AFC_West": ["KC", "LAC", "DEN", "LV"],
    "NFC_East": ["DAL", "NYG", "PHI", "WAS"],
    "NFC_North": ["CHI", "DET", "GB", "MIN"],
    "NFC_South": ["ATL", "CAR", "NO", "TB"],
    "NFC_West": ["ARI", "LAR", "SEA", "SF"],
}

# NBA conference structure (simplified)
NBA_CONFERENCES: dict[str, list[str]] = {
    "East": [
        "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND",
        "MIA", "MIL", "NYK", "ORL", "PHI", "TOR", "WAS",
    ],
    "West": [
        "DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN",
        "NOP", "OKC", "PHX", "POR", "SAC", "SAS", "UTA",
    ],
}


class SeasonSimulator:
    """Monte Carlo season simulation engine."""

    def __init__(self, n_simulations: int = 10000, seed: int | None = None):
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)

    def simulate_nfl_season(
        self,
        schedule: pd.DataFrame,
        win_probs: dict[str, float] | None = None,
        get_win_prob: callable | None = None,
    ) -> SimulationResult:
        """Simulate a full NFL season.

        Args:
            schedule: DataFrame with home_team, away_team columns
            win_probs: Dict mapping game_id -> home win probability
            get_win_prob: Callable(home_team, away_team) -> home_win_prob
                         Used if win_probs is None
        """
        teams = sorted(
            set(schedule["home_team"].unique()) | set(schedule["away_team"].unique())
        )
        team_idx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)

        # Pre-compute win probabilities for each game
        game_probs = []
        for _, game in schedule.iterrows():
            home = game["home_team"]
            away = game["away_team"]
            game_id = game.get("game_id", f"{home}_{away}")

            if win_probs and game_id in win_probs:
                prob = win_probs[game_id]
            elif get_win_prob:
                prob = get_win_prob(home, away)
            else:
                prob = 0.55  # default slight home advantage

            game_probs.append((team_idx[home], team_idx[away], prob))

        # Run simulations
        wins_matrix = np.zeros((self.n_simulations, n_teams), dtype=np.int32)

        for sim in range(self.n_simulations):
            for home_idx, away_idx, prob in game_probs:
                if self.rng.random() < prob:
                    wins_matrix[sim, home_idx] += 1
                else:
                    wins_matrix[sim, away_idx] += 1

        # Determine playoffs for each simulation
        playoff_counts = np.zeros(n_teams, dtype=np.int32)
        div_winner_counts = np.zeros(n_teams, dtype=np.int32)

        for sim in range(self.n_simulations):
            sim_wins = wins_matrix[sim]

            # Division winners
            for div_name, div_teams in NFL_DIVISIONS.items():
                div_indices = [team_idx[t] for t in div_teams if t in team_idx]
                if not div_indices:
                    continue
                div_wins = [(idx, sim_wins[idx]) for idx in div_indices]
                # Break ties randomly
                self.rng.shuffle(div_wins)
                winner_idx = max(div_wins, key=lambda x: x[1])[0]
                div_winner_counts[winner_idx] += 1
                playoff_counts[winner_idx] += 1

            # Wild cards: top 3 non-division-winners per conference
            for conf in ["AFC", "NFC"]:
                conf_divs = [d for d in NFL_DIVISIONS if d.startswith(conf)]
                conf_teams = []
                for div_name in conf_divs:
                    for t in NFL_DIVISIONS[div_name]:
                        if t in team_idx:
                            conf_teams.append(team_idx[t])

                conf_div_winners = set()
                for div_name in conf_divs:
                    div_indices = [team_idx[t] for t in NFL_DIVISIONS[div_name] if t in team_idx]
                    if div_indices:
                        winner = max(div_indices, key=lambda i: sim_wins[i])
                        conf_div_winners.add(winner)

                wild_card_pool = [
                    (idx, sim_wins[idx])
                    for idx in conf_teams
                    if idx not in conf_div_winners
                ]
                wild_card_pool.sort(key=lambda x: x[1], reverse=True)
                for idx, _ in wild_card_pool[:3]:
                    playoff_counts[idx] += 1

        # Build outcomes
        outcomes = []
        for i, team in enumerate(teams):
            team_wins = wins_matrix[:, i]

            # Win distribution
            win_counts = np.bincount(team_wins, minlength=18)
            win_dist = {w: win_counts[w] / self.n_simulations for w in range(18)}

            outcomes.append(SeasonOutcome(
                team=team,
                mean_wins=float(team_wins.mean()),
                median_wins=float(np.median(team_wins)),
                std_wins=float(team_wins.std()),
                win_distribution=win_dist,
                playoff_prob=playoff_counts[i] / self.n_simulations,
                division_winner_prob=div_winner_counts[i] / self.n_simulations,
                conference_winner_prob=0.0,  # Would need playoff sim
                champion_prob=0.0,
            ))

        return SimulationResult(
            n_simulations=self.n_simulations,
            team_outcomes=sorted(outcomes, key=lambda o: o.mean_wins, reverse=True),
            raw_wins=wins_matrix,
        )

    def simulate_nba_season(
        self,
        schedule: pd.DataFrame,
        get_win_prob: callable | None = None,
    ) -> SimulationResult:
        """Simulate a full NBA season."""
        teams = sorted(
            set(schedule["home_team"].unique()) | set(schedule["away_team"].unique())
        )
        team_idx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)

        game_probs = []
        for _, game in schedule.iterrows():
            home = game["home_team"]
            away = game["away_team"]
            if get_win_prob:
                prob = get_win_prob(home, away)
            else:
                prob = 0.6  # NBA home advantage is larger
            game_probs.append((team_idx[home], team_idx[away], prob))

        wins_matrix = np.zeros((self.n_simulations, n_teams), dtype=np.int32)

        for sim in range(self.n_simulations):
            for home_idx, away_idx, prob in game_probs:
                if self.rng.random() < prob:
                    wins_matrix[sim, home_idx] += 1
                else:
                    wins_matrix[sim, away_idx] += 1

        # Playoff determination (top 6 + play-in 7-10)
        playoff_counts = np.zeros(n_teams, dtype=np.int32)

        for sim in range(self.n_simulations):
            sim_wins = wins_matrix[sim]
            for conf, conf_teams in NBA_CONFERENCES.items():
                conf_indices = [team_idx[t] for t in conf_teams if t in team_idx]
                conf_wins = [(idx, sim_wins[idx]) for idx in conf_indices]
                conf_wins.sort(key=lambda x: x[1], reverse=True)
                # Top 10 make playoffs (including play-in)
                for idx, _ in conf_wins[:10]:
                    playoff_counts[idx] += 1

        outcomes = []
        for i, team in enumerate(teams):
            team_wins = wins_matrix[:, i]
            win_counts = np.bincount(team_wins, minlength=83)
            win_dist = {w: win_counts[w] / self.n_simulations for w in range(83)}

            outcomes.append(SeasonOutcome(
                team=team,
                mean_wins=float(team_wins.mean()),
                median_wins=float(np.median(team_wins)),
                std_wins=float(team_wins.std()),
                win_distribution=win_dist,
                playoff_prob=playoff_counts[i] / self.n_simulations,
                division_winner_prob=0.0,
                conference_winner_prob=0.0,
                champion_prob=0.0,
            ))

        return SimulationResult(
            n_simulations=self.n_simulations,
            team_outcomes=sorted(outcomes, key=lambda o: o.mean_wins, reverse=True),
            raw_wins=wins_matrix,
        )

    def compute_win_total_probs(
        self, outcome: SeasonOutcome, line: float
    ) -> tuple[float, float]:
        """Compute over/under probability for a win total line.

        Returns (over_prob, under_prob).
        """
        over = sum(
            prob for wins, prob in outcome.win_distribution.items()
            if wins > line
        )
        under = sum(
            prob for wins, prob in outcome.win_distribution.items()
            if wins < line
        )
        push = sum(
            prob for wins, prob in outcome.win_distribution.items()
            if wins == line
        )
        # Half the push probability goes to each side
        return over + push / 2, under + push / 2

# Sports Prediction Market System

A comprehensive sports prediction market system covering **NFL**, **NBA**, and **MLB** for season-long and game-level markets on Kalshi and Polymarket.

## Features

- **Game-level predictions** — Elo + calibrated efficiency model → win probabilities, spreads, and totals
- **Player prop modeling** — Hierarchical Bayesian model for stat distributions with empirical Bayes shrinkage
- **Season simulation** — Monte Carlo engine (10,000+ iterations) for win totals, playoff odds, and futures
- **Edge detection** — Model vs market price comparison with Kelly criterion position sizing
- **Market integration** — Live prices from Kalshi and Polymarket APIs
- **Portfolio management** — Position tracking, exposure limits, and P&L reporting
- **Weather impact** — NOAA forecasts for outdoor NFL and MLB games

## Installation on macOS

### Prerequisites

- Python 3.10 or later
- Git
- pip (comes with Python)

### Step 1: Install Python (if needed)

Using [Homebrew](https://brew.sh/):

```bash
brew install python@3.12
```

Or download directly from [python.org](https://www.python.org/downloads/macos/).

Verify your installation:

```bash
python3 --version  # Should be 3.10+
```

### Step 2: Clone the repository

```bash
git clone https://github.com/powoso/Claude-sports.git
cd Claude-sports
```

### Step 3: Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 4: Install the package

```bash
pip install -e ".[dev]"
```

This installs all dependencies including:

| Package | Purpose |
|---|---|
| `nfl-data-py` | NFL play-by-play and schedule data |
| `nba_api` | NBA game logs and advanced stats |
| `pybaseball` | MLB Statcast and FanGraphs data |
| `pymc` | Bayesian modeling (optional, for full MCMC) |
| `scikit-learn` | Calibrated classification models |
| `scipy` | Statistical distributions |
| `rich` | Terminal UI formatting |
| `httpx` | Async-capable HTTP client for market APIs |

### Step 5: Verify installation

```bash
# Run the test suite
pytest

# Check the CLI
sports-predict --help
```

## Usage

### Game Predictions

```bash
# NFL week predictions
sports-predict predict nfl --season 2025 --week 10

# NBA predictions for a specific date
sports-predict predict nba --season 2025-26 --date 2026-01-15

# JSON output for programmatic use
sports-predict predict nfl --season 2025 --week 10 --json-output
```

### Season Simulation

```bash
# Monte Carlo season simulation (10,000 iterations)
sports-predict simulate nfl
sports-predict simulate nba
```

### Elo Rankings

```bash
sports-predict rankings nfl --season 2025
sports-predict rankings nba --season 2025-26
```

### Market Prices

```bash
# Fetch live prediction market prices
sports-predict markets nfl
sports-predict markets nba
sports-predict markets mlb
```

### Portfolio

```bash
# View portfolio status and P&L
sports-predict portfolio
```

### Using as a Library

```python
from sports_predict.engine import PredictionEngine

engine = PredictionEngine()

# NFL game predictions with edge detection
result = engine.run_nfl_predictions(season=2025, week=10)
for game in result["games"]:
    print(f"{game['away_team']} @ {game['home_team']}: {game['home_win_prob']:.1%}")

for edge in result["edges"]:
    print(f"  EDGE: {edge['event']} — {edge['edge_pct']:+.1f}% (Kelly: {edge['kelly_size']:.1f}%)")

# Season simulation
sim = engine.run_season_simulation("nfl")
for team in sim["team_outcomes"][:10]:
    print(f"{team['team']}: {team['mean_wins']:.1f} wins, {team['playoff_prob']:.1%} playoff")

# Player props
from sports_predict.models.player_props import PlayerPropModel
import pandas as pd

model = PlayerPropModel(sport="nfl")
# ... fit with game log data, then predict
pred = model.predict_prop("Patrick Mahomes", "passing_yards", 275.5)
print(f"Over {pred.line}: {pred.over_prob:.1%} | Projected: {pred.projected_value}")
```

## Configuration

Edit `config/settings.yaml` to customize:

```yaml
nfl:
  elo_k_factor: 20          # Elo update speed
  elo_home_advantage: 48     # Home field in Elo points
  epa_smoothing_window: 4    # Rolling average window (games)

models:
  monte_carlo_iterations: 10000

edge_detection:
  min_edge_pct: 3.0          # Minimum edge to flag
  kelly_fraction: 0.25       # Quarter-Kelly sizing
  max_position_pct: 5.0      # Max single position (% of bankroll)
```

### Environment Variables

Set API keys for authenticated market access:

```bash
export KALSHI_API_KEY="your-key"
export POLYMARKET_API_KEY="your-key"
export NOAA_API_TOKEN="your-token"
```

## Architecture

```
sports_predict/
├── data_collection/        # Data ingestion
│   ├── nfl.py             # Play-by-play, EPA, schedules, injuries
│   ├── nba.py             # Game logs, net rating, rest/travel
│   ├── mlb.py             # Statcast, wOBA, park factors
│   ├── odds.py            # Kalshi + Polymarket APIs
│   └── weather.py         # NOAA forecasts for outdoor stadiums
├── features/               # Feature engineering
│   ├── nfl_features.py    # Rolling EPA, CPOE, red zone, situational
│   ├── nba_features.py    # Net rating, B2B penalties, travel fatigue
│   └── mlb_features.py    # Park-adjusted wOBA, FIP, bullpen quality
├── models/                 # Prediction models
│   ├── elo.py             # Sport-tuned Elo with MOV adjustment
│   ├── efficiency.py      # Ridge + isotonic calibration → win probs
│   ├── player_props.py    # Empirical Bayes hierarchical model
│   ├── season_sim.py      # Monte Carlo with playoff structure
│   └── totals.py          # Pace × efficiency → score distributions
├── edge_detection/
│   └── edge_finder.py     # Kelly sizing, bias detection, line tracking
├── market/
│   └── portfolio.py       # Position management, exposure limits, P&L
├── engine.py               # Orchestration layer
└── cli.py                  # Rich CLI interface
```

### Model Pipeline

```
Data Sources → Feature Engineering → [Elo + Efficiency Model] → Calibrated Win Prob
                                          ↓                           ↓
                                   Player Prop Model          Edge Detection
                                          ↓                     ↓         ↓
                                   Season Simulator      Kelly Sizing  Bias Alerts
                                          ↓                     ↓
                                   Win Totals/Futures    Portfolio Manager
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=sports_predict

# Specific test module
pytest tests/test_elo.py -v
```

## Uploading to GitHub

### First-time setup

```bash
# Create a new repository on GitHub (via browser or gh CLI)
gh repo create Claude-sports --public --source=. --push

# Or manually:
git remote add origin https://github.com/YOUR_USERNAME/Claude-sports.git
git push -u origin claude/sports-prediction-markets-UIzsJ
```

### Creating a pull request

```bash
gh pr create --title "Sports prediction market system" --body "Complete NFL/NBA/MLB prediction pipeline with edge detection"
```

### Subsequent pushes

```bash
git add -A
git commit -m "Your commit message"
git push
```

## License

MIT

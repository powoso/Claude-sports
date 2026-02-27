"""CLI entry point for the sports prediction market system."""

from __future__ import annotations

import json
import logging
import sys

import click
from rich.console import Console
from rich.table import Table

from sports_predict.engine import PredictionEngine
from sports_predict.utils.config import load_settings

console = Console()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.option("--config", "-c", type=click.Path(), default=None, help="Config file path")
@click.pass_context
def main(ctx, verbose: bool, config: str | None):
    """Sports Prediction Market System - NFL, NBA, MLB."""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    from pathlib import Path
    settings = load_settings(Path(config) if config else None)
    ctx.obj["settings"] = settings
    ctx.obj["engine"] = PredictionEngine(settings)


@main.command()
@click.argument("sport", type=click.Choice(["nfl", "nba", "mlb"]))
@click.option("--season", "-s", type=str, required=True, help="Season (e.g., 2024, 2024-25)")
@click.option("--week", "-w", type=int, default=None, help="NFL week number")
@click.option("--date", "-d", type=str, default=None, help="Game date (YYYY-MM-DD)")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.pass_context
def predict(ctx, sport: str, season: str, week: int | None, date: str | None, json_output: bool):
    """Generate game predictions for a sport."""
    engine: PredictionEngine = ctx.obj["engine"]

    if sport == "nfl":
        if week is None:
            click.echo("Error: --week is required for NFL predictions", err=True)
            sys.exit(1)
        result = engine.run_nfl_predictions(int(season), week)
    elif sport == "nba":
        result = engine.run_nba_predictions(season, date)
    else:
        click.echo(f"Predictions not yet implemented for {sport}", err=True)
        sys.exit(1)

    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
        return

    # Rich table output
    games = result.get("games", [])
    if not games:
        console.print("[yellow]No games found for the specified criteria.[/yellow]")
        return

    table = Table(title=f"{sport.upper()} Predictions")
    table.add_column("Matchup", style="bold")
    table.add_column("Home Win %", justify="right")
    table.add_column("Spread", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Conf", justify="right")

    for g in games:
        wp = g["home_win_prob"]
        wp_color = "green" if wp > 0.55 else "red" if wp < 0.45 else "yellow"
        table.add_row(
            f"{g['away_team']} @ {g['home_team']}",
            f"[{wp_color}]{wp:.1%}[/{wp_color}]",
            f"{g['projected_spread']:+.1f}",
            f"{g['projected_total']:.1f}",
            f"{g.get('projected_away_score', '?')}-{g.get('projected_home_score', '?')}",
            f"{g['confidence']:.0%}",
        )
    console.print(table)

    # Show edges
    edges = result.get("edges", [])
    if edges:
        edge_table = Table(title="Detected Edges")
        edge_table.add_column("Event")
        edge_table.add_column("Outcome")
        edge_table.add_column("Model", justify="right")
        edge_table.add_column("Market", justify="right")
        edge_table.add_column("Edge %", justify="right")
        edge_table.add_column("Kelly %", justify="right")
        edge_table.add_column("Source")

        for e in edges:
            edge_color = "green" if e["edge_pct"] > 0 else "red"
            edge_table.add_row(
                e["event"],
                e["outcome"],
                f"{e['model_prob']:.1%}",
                f"{e['market_prob']:.1%}",
                f"[{edge_color}]{e['edge_pct']:+.1f}%[/{edge_color}]",
                f"{e['kelly_size']:.1f}%",
                e["source"],
            )
        console.print(edge_table)
    else:
        console.print("[dim]No actionable edges detected.[/dim]")


@main.command()
@click.argument("sport", type=click.Choice(["nfl", "nba", "mlb"]))
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.pass_context
def simulate(ctx, sport: str, json_output: bool):
    """Run Monte Carlo season simulation."""
    engine: PredictionEngine = ctx.obj["engine"]
    result = engine.run_season_simulation(sport)

    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
        return

    outcomes = result.get("team_outcomes", [])
    if not outcomes:
        console.print(f"[red]{result.get('error', 'No results')}[/red]")
        return

    table = Table(
        title=f"{sport.upper()} Season Simulation ({result['n_simulations']:,} iterations)"
    )
    table.add_column("Team", style="bold")
    table.add_column("Avg Wins", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Std Dev", justify="right")
    table.add_column("Playoff %", justify="right")
    table.add_column("Div Winner %", justify="right")

    for o in outcomes[:32]:
        playoff_color = "green" if o["playoff_prob"] > 0.5 else "yellow" if o["playoff_prob"] > 0.2 else "red"
        table.add_row(
            o["team"],
            f"{o['mean_wins']:.1f}",
            f"{o['median_wins']:.0f}",
            f"{o['std_wins']:.1f}",
            f"[{playoff_color}]{o['playoff_prob']:.1%}[/{playoff_color}]",
            f"{o['division_winner_prob']:.1%}",
        )
    console.print(table)


@main.command()
@click.argument("sport", type=click.Choice(["nfl", "nba", "mlb"]))
@click.option("--season", "-s", type=str, required=True)
@click.pass_context
def rankings(ctx, sport: str, season: str):
    """Show current Elo rankings."""
    engine: PredictionEngine = ctx.obj["engine"]

    sport_engine = {"nfl": engine.nfl, "nba": engine.nba, "mlb": engine.mlb}[sport]
    elo_rankings = sport_engine.elo.get_rankings()

    if elo_rankings.empty:
        console.print("[yellow]No Elo ratings available. Run predictions first to build ratings.[/yellow]")
        return

    table = Table(title=f"{sport.upper()} Elo Rankings")
    table.add_column("Rank", justify="right")
    table.add_column("Team", style="bold")
    table.add_column("Elo", justify="right")
    table.add_column("Record", justify="right")
    table.add_column("Win %", justify="right")

    for i, (_, row) in enumerate(elo_rankings.iterrows(), 1):
        table.add_row(
            str(i),
            row["team"],
            f"{row['elo']:.0f}",
            f"{row['wins']}-{row['losses']}",
            f"{row['win_pct']:.3f}",
        )
    console.print(table)


@main.command()
@click.argument("sport", type=click.Choice(["nfl", "nba", "mlb"]))
@click.pass_context
def markets(ctx, sport: str):
    """Fetch and display current prediction market prices."""
    engine: PredictionEngine = ctx.obj["engine"]
    from sports_predict.data_collection.odds import Sport

    sport_enum = {"nfl": Sport.NFL, "nba": Sport.NBA, "mlb": Sport.MLB}[sport]
    prices = engine.odds.get_all_market_prices(sport_enum)

    if prices.empty:
        console.print("[yellow]No market prices available.[/yellow]")
        return

    table = Table(title=f"{sport.upper()} Market Prices")
    table.add_column("Event")
    table.add_column("Outcome")
    table.add_column("Price", justify="right")
    table.add_column("Source")

    for _, row in prices.head(50).iterrows():
        table.add_row(
            str(row.get("event_name", ""))[:60],
            str(row.get("outcome", "")),
            f"{row['price']:.1%}",
            str(row.get("source", "")),
        )
    console.print(table)


@main.command()
@click.pass_context
def portfolio(ctx):
    """Show current portfolio status."""
    engine: PredictionEngine = ctx.obj["engine"]
    summary = engine.get_portfolio_summary()

    table = Table(title="Portfolio Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Bankroll", f"${summary['bankroll']:,.2f}")
    table.add_row("Active Positions", str(summary["active_positions"]))
    table.add_row("Total Exposure", f"${summary['total_exposure']:,.2f}")
    table.add_row("Exposure %", f"{summary['exposure_pct']:.1f}%")
    table.add_row("Unrealized P&L", f"${summary['unrealized_pnl']:,.2f}")
    table.add_row("Realized P&L", f"${summary['realized_pnl']:,.2f}")
    table.add_row("Total P&L", f"${summary['total_pnl']:,.2f}")
    table.add_row("ROI", f"{summary['roi_pct']:.2f}%")
    table.add_row("Win Rate", f"{summary['win_rate']:.1%}")
    table.add_row("Resolved Bets", str(summary["total_resolved"]))

    console.print(table)

    # Sport breakdown
    by_sport = summary.get("by_sport", {})
    if by_sport:
        sport_table = Table(title="By Sport")
        sport_table.add_column("Sport")
        sport_table.add_column("Positions", justify="right")
        sport_table.add_column("Exposure", justify="right")
        sport_table.add_column("P&L", justify="right")

        for sport_name, data in by_sport.items():
            sport_table.add_row(
                sport_name.upper(),
                str(data["positions"]),
                f"${data['exposure']:,.2f}",
                f"${data['unrealized_pnl']:,.2f}",
            )
        console.print(sport_table)


@main.command()
@click.pass_context
def clear_cache(ctx):
    """Clear the data cache."""
    engine: PredictionEngine = ctx.obj["engine"]
    count = engine.cache.clear_expired()
    console.print(f"Removed {count} expired cache entries.")


if __name__ == "__main__":
    main()

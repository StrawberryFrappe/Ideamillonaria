"""Command line interface: ``ideam --help``."""

from __future__ import annotations

import warnings
from typing import List, Optional

import pandas as pd
import typer

from . import config, journal, pipeline
from .config import LEAGUES, get_league

warnings.filterwarnings("ignore", category=UserWarning)
app = typer.Typer(add_completion=False, help="Ideamillonaria - football prediction & decision support")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def _codes(leagues: List[str]) -> List[str]:
    if not leagues or leagues == ["all"]:
        return [c for c in LEAGUES if c != "SYN"]
    return [get_league(c).code for c in leagues]


@app.command()
def leagues():
    """List supported leagues."""
    for lg in LEAGUES.values():
        typer.echo(f"{lg.code:12s} {lg.country:12s} {lg.name:28s} source={lg.source}:{lg.source_id}")


@app.command()
def fetch(leagues: List[str] = typer.Argument(..., help="League codes, e.g. EPL CHI1 INT"),
          full: bool = typer.Option(False, help="Re-download everything")):
    """Download / refresh match data (results + fixtures + odds where available)."""
    from .data.fetch import update_league
    for code in _codes(leagues):
        df = update_league(get_league(code), full=full)
        played = df["hg"].notna().sum()
        typer.echo(f"{code}: {played} results, {len(df) - played} upcoming fixtures, "
                   f"latest result {df.loc[df['hg'].notna(), 'date'].max()}")


@app.command()
def train(leagues: List[str] = typer.Argument(...)):
    """Walk-forward backtest + train production models + predict upcoming fixtures."""
    for code in _codes(leagues):
        lg = get_league(code)
        art, prep = pipeline.train(lg)
        pipeline.predict(lg, art, prep)


@app.command()
def backtest(league: str, min_edge: float = 0.05, max_odds: float = 7.0):
    """Show scoring-rule metrics and value-betting backtests per model view."""
    from .evaluate import betting, metrics
    o = pipeline.outputs(league)
    if not o["backtest"].exists():
        typer.echo("No backtest yet - run: ideam train " + league)
        raise typer.Exit(1)
    bt = pd.read_parquet(o["backtest"])
    views = [v for v in ["base", "elo", "dc", "gbm", "mkt", "ens", "book_open", "book_close"] if f"p_{v}_h" in bt]
    typer.echo("\n== Accuracy (lower RPS / log-loss is better) - each view on the matches it covers ==")
    typer.echo(metrics.metrics_table(bt, views)[["n", "rps", "logloss", "brier", "accuracy"]].round(4).to_string())
    with_odds = bt.dropna(subset=["p_book_open_h"])
    if len(with_odds):
        typer.echo("\n== Same, restricted to matches with bookmaker odds ==")
        typer.echo(metrics.metrics_table(with_odds, views, common=True)[["n", "rps", "logloss", "accuracy"]]
                   .round(4).to_string())
        typer.echo(f"\n== Value-bet backtest (bet at opening odds when edge >= {min_edge:.0%}, odds <= {max_odds}) ==")
        res = betting.simulate_all(bt, ["dc", "gbm", "mkt", "ens"], min_edge=min_edge, max_odds=max_odds)
        typer.echo(res.round(3).to_string())
        typer.echo("\nroi_ci = 95% bootstrap interval. If it includes 0, the 'profit' may be pure luck.")
        typer.echo("clv > 0 means bets beat the closing line - the most reliable sign of a real edge.")
    else:
        typer.echo("\n(No bookmaker odds in this league's history - betting backtest unavailable.)")


@app.command()
def predict(leagues: List[str] = typer.Argument(...)):
    """Predict upcoming fixtures with the saved models."""
    for code in _codes(leagues):
        typer.echo(f"\n=== {code} ===")
        pipeline.predict(get_league(code))


@app.command()
def matchup(league: str, home: str, away: str, date: Optional[str] = None,
            neutral: bool = False):
    """Predict any matchup, e.g. ideam matchup INT Chile Peru --neutral"""
    from .data import store
    from .report import markets
    lg = get_league(league)
    art = pipeline.load_artifacts(lg.code)
    p, _ = pipeline.predict_custom(lg, art, store.load_matches(lg.code), home, away, date, neutral)
    r = p.iloc[0]
    typer.echo(f"{home} vs {away}")
    for v in ["elo", "dc", "gbm", "ens"]:
        typer.echo(f"  {v:4s}  H {r[f'p_{v}_h']:.1%}  D {r[f'p_{v}_d']:.1%}  A {r[f'p_{v}_a']:.1%}")
    typer.echo(f"  agreement: {r['agreement']}")
    if art.dc_fit is not None:
        typer.echo(markets.top_scorelines(r["dc_lambda_h"], r["dc_lambda_a"], art.dc_fit.rho, 5).round(3).to_string())


@app.command("snapshot-odds")
def snapshot_odds(leagues: List[str] = typer.Argument(...)):
    """Save current bookmaker odds (needs ODDS_API_KEY). Run regularly to build odds history."""
    from .data.fetch import snapshot_odds as snap
    for code in _codes(leagues):
        try:
            typer.echo(f"{code}: stored {snap(get_league(code))} bookmaker prices")
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"{code}: {exc}")


@app.command()
def update(leagues: List[str] = typer.Argument(..., help="e.g. CHI1 EPL"),
           odds: bool = typer.Option(True, help="Also snapshot odds if ODDS_API_KEY is set")):
    """Daily routine: fetch -> odds snapshot -> retrain -> predict -> settle journal."""
    from .data.fetch import snapshot_odds as snap
    from .data.fetch import update_league
    for code in _codes(leagues):
        lg = get_league(code)
        typer.echo(f"\n=== {code} ===")
        if odds and config.ODDS_API_KEY and lg.odds_api_hint:
            try:
                typer.echo(f"odds snapshot: {snap(lg)} prices")
            except Exception as exc:  # noqa: BLE001
                typer.echo(f"odds snapshot failed: {exc}")
        df = update_league(lg)
        settled = journal.settle(df, lg.code)
        if settled:
            typer.echo(f"settled {settled} journal bets")
        art, prep = pipeline.train(lg)
        pipeline.predict(lg, art, prep)


@app.command()
def doctor():
    """Check which data sources are reachable from this machine."""
    import requests
    checks = {
        "football-data.co.uk": "https://www.football-data.co.uk/fixtures.csv",
        "ESPN (Chile)": "https://site.api.espn.com/apis/site/v2/sports/soccer/chi.1/scoreboard",
        "martj42 (internationals)": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
        "football-data.org": "https://api.football-data.org/v4/competitions",
        "The Odds API": "https://api.the-odds-api.com/v4/sports",
    }
    for name, url in checks.items():
        try:
            r = requests.head(url, timeout=15, allow_redirects=True, headers=config.HTTP_HEADERS)
            status = r.status_code
            ok = status < 500 and status not in (403, 407)
            typer.echo(f"{'OK ' if ok else 'ERR'} {name:28s} HTTP {status}")
        except requests.RequestException as exc:
            typer.echo(f"ERR {name:28s} {type(exc).__name__}")
    typer.echo(f"\nODDS_API_KEY set: {bool(config.ODDS_API_KEY)}   FOOTBALL_DATA_ORG_KEY set: "
               f"{bool(config.FOOTBALL_DATA_ORG_KEY)}")
    typer.echo(f"Data directory: {config.DATA_DIR}")


@app.command()
def demo():
    """Offline demo on a synthetic league with known truth (no internet needed)."""
    pipeline.run("SYN", fetch=True)
    backtest("SYN")


@app.command()
def dashboard():
    """Open the web dashboard."""
    import subprocess
    import sys
    from pathlib import Path
    app_path = Path(__file__).resolve().parents[2] / "app" / "dashboard.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=False)


if __name__ == "__main__":
    app()

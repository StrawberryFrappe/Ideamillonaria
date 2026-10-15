"""League registry, paths and settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


def _load_dotenv() -> None:
    """Minimal .env loader (KEY=VALUE lines) so we don't need python-dotenv."""
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if candidate.is_file():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            break


_load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA = (_REPO_ROOT if (_REPO_ROOT / "pyproject.toml").exists() else Path.cwd()) / "data"
DATA_DIR = Path(os.environ.get("IDEAM_DATA_DIR", _DEFAULT_DATA)).resolve()
RAW_DIR = DATA_DIR / "raw"
MATCHES_DIR = DATA_DIR / "matches"
ODDS_DIR = DATA_DIR / "odds_snapshots"
MODELS_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "output"
JOURNAL_PATH = DATA_DIR / "journal.sqlite"

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
FOOTBALL_DATA_ORG_KEY = os.environ.get("FOOTBALL_DATA_ORG_KEY", "")
GBM_DEVICE = os.environ.get("IDEAM_GBM_DEVICE", "cpu")

HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (ideamillonaria football research tool)"}


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    home_adv: float = 65.0
    season_regress: float = 0.20  # pull ratings this far back to the mean between seasons
    newcomer_offset: float = -80.0  # promoted / new teams start below the league mean


@dataclass(frozen=True)
class DCParams:
    xi: float = 0.0019  # time decay per day (half-life ~ 1 year)
    window_days: int = 1460  # only matches from the last N days enter a fit
    refit_days: int = 7  # walk-forward refit frequency
    ridge: float = 2.0  # Gaussian prior precision on attack/defence (shrinks small samples to average)


@dataclass(frozen=True)
class League:
    code: str
    name: str
    country: str
    source: str  # football_data | football_data_new | espn | international
    source_id: str
    season_type: str = "split"  # split (Aug-May) | calendar (Jan-Dec)
    first_season: int = 2005
    odds_api_hint: str | None = None  # substring used to discover the-odds-api sport key
    elo: EloParams = field(default_factory=EloParams)
    dc: DCParams = field(default_factory=DCParams)
    min_train_seasons: int = 3  # seasons of history before the first backtested season
    backtest_from: int | None = None  # first season (start year) to backtest

    def season_of(self, date: pd.Timestamp) -> str:
        if self.season_type == "calendar":
            return str(date.year)
        start = date.year if date.month >= 7 else date.year - 1
        return f"{start}-{str(start + 1)[-2:]}"

    def seasons_of(self, dates: pd.Series) -> pd.Series:
        dates = pd.to_datetime(dates)
        if self.season_type == "calendar":
            return dates.dt.year.astype(str)
        start = dates.dt.year.where(dates.dt.month >= 7, dates.dt.year - 1)
        return start.astype(str) + "-" + (start + 1).astype(str).str[-2:]


def _fd(code, name, country, div, hint, first=2005):
    return League(code, name, country, "football_data", div, "split", first, hint)


LEAGUES: dict[str, League] = {
    lg.code: lg
    for lg in [
        # --- Europe (football-data.co.uk: results, shots, opening + closing odds) ---
        _fd("EPL", "Premier League", "England", "E0", "soccer_epl"),
        _fd("ELC", "Championship", "England", "E1", "soccer_efl_champ"),
        _fd("LALIGA", "La Liga", "Spain", "SP1", "soccer_spain_la_liga"),
        _fd("LALIGA2", "La Liga 2", "Spain", "SP2", "soccer_spain_segunda_division"),
        _fd("SERIEA", "Serie A", "Italy", "I1", "soccer_italy_serie_a"),
        _fd("SERIEB", "Serie B", "Italy", "I2", "soccer_italy_serie_b"),
        _fd("BUNDESLIGA", "Bundesliga", "Germany", "D1", "soccer_germany_bundesliga"),
        _fd("BUNDESLIGA2", "2. Bundesliga", "Germany", "D2", "soccer_germany_bundesliga2"),
        _fd("LIGUE1", "Ligue 1", "France", "F1", "soccer_france_ligue_one"),
        _fd("LIGUE2", "Ligue 2", "France", "F2", "soccer_france_ligue_two"),
        _fd("EREDIVISIE", "Eredivisie", "Netherlands", "N1", "soccer_netherlands_eredivisie"),
        _fd("PRIMEIRA", "Primeira Liga", "Portugal", "P1", "soccer_portugal_primeira_liga"),
        # --- Americas, football-data.co.uk "extra leagues" (results + closing odds) ---
        League("ARG", "Liga Profesional", "Argentina", "football_data_new", "ARG", "calendar", 2012,
               "soccer_argentina_primera_division", min_train_seasons=2),
        League("BRA", "Serie A", "Brazil", "football_data_new", "BRA", "calendar", 2012,
               "soccer_brazil_campeonato", min_train_seasons=2),
        League("MEX", "Liga MX", "Mexico", "football_data_new", "MEX", "split", 2012,
               "soccer_mexico_ligamx", min_train_seasons=2),
        # --- Chile (ESPN public JSON: results, fixtures, match stats, odds when listed) ---
        League("CHI1", "Primera División", "Chile", "espn", "chi.1", "calendar", 2012, "chile",
               elo=EloParams(k=24.0, home_adv=75.0, season_regress=0.25, newcomer_offset=-90.0),
               dc=DCParams(xi=0.0025, window_days=1100, refit_days=7, ridge=4.0),
               min_train_seasons=3),
        League("CHI2", "Primera B", "Chile", "espn", "chi.2", "calendar", 2014, None,
               elo=EloParams(k=24.0, home_adv=75.0, season_regress=0.30, newcomer_offset=-60.0),
               dc=DCParams(xi=0.0025, window_days=1100, refit_days=7, ridge=5.0),
               min_train_seasons=3),
        # --- National teams (martj42/international_results) ---
        League("INT", "International matches", "World", "international", "martj42", "calendar", 1872,
               None,
               elo=EloParams(k=40.0, home_adv=90.0, season_regress=0.0, newcomer_offset=-150.0),
               dc=DCParams(xi=0.0012, window_days=2920, refit_days=30, ridge=0.5),
               min_train_seasons=6, backtest_from=2014),
        # --- Synthetic league with known truth (offline demo / tests) ---
        League("SYN", "Synthetic demo league", "Nowhere", "synthetic", "synthetic", "split", 2015, None,
               min_train_seasons=3),
    ]
}


def get_league(code: str) -> League:
    try:
        return LEAGUES[code.upper()]
    except KeyError as exc:
        raise KeyError(f"Unknown league '{code}'. Known: {', '.join(LEAGUES)}") from exc


def ensure_dirs() -> None:
    for d in (RAW_DIR, MATCHES_DIR, ODDS_DIR, MODELS_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

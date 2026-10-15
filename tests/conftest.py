import warnings
from pathlib import Path

import pytest

from ideamillonaria import config

warnings.filterwarnings("ignore", category=UserWarning)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    return FIXTURES


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    """Point every data path at a temporary directory."""
    for name, sub in [("DATA_DIR", ""), ("RAW_DIR", "raw"), ("MATCHES_DIR", "matches"),
                      ("ODDS_DIR", "odds_snapshots"), ("MODELS_DIR", "models"), ("OUTPUT_DIR", "output")]:
        monkeypatch.setattr(config, name, tmp_path / sub if sub else tmp_path)
    monkeypatch.setattr(config, "JOURNAL_PATH", tmp_path / "journal.sqlite")
    config.ensure_dirs()
    return tmp_path


@pytest.fixture(scope="session")
def synthetic():
    from ideamillonaria.synthetic import generate_league
    return generate_league(n_teams=12, seasons=6, seed=3)

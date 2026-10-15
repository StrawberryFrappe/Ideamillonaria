"""Parsers tested on small hand-written payloads that follow each provider's format."""

import json

import numpy as np
import pandas as pd
import pytest

from ideamillonaria.config import get_league
from ideamillonaria.data import teams
from ideamillonaria.data.sources import espn, football_data, international, odds_api


def test_football_data_main(fixtures_dir):
    df = football_data.parse_main((fixtures_dir / "football_data_E0.csv").read_text(), get_league("EPL"))
    assert len(df) == 2
    r = df.iloc[0]
    assert (r["home"], r["away"], r["hg"], r["ag"]) == ("Man United", "Fulham", 1, 0)
    assert r["season"] == "2024-25"
    assert r["h_shots"] == 14 and r["a_sot"] == 2
    assert r["odds_h_open"] == pytest.approx(1.6)  # Avg preferred over B365/PS
    assert r["odds_h_close"] == pytest.approx(1.64)
    assert r["odds_o25_open"] == pytest.approx(1.73)


def test_football_data_new(fixtures_dir):
    df = football_data.parse_new((fixtures_dir / "football_data_new_ARG.csv").read_text(), get_league("ARG"))
    assert len(df) == 1 and df.iloc[0]["season"] == "2024"
    assert df.iloc[0]["odds_h_close"] == pytest.approx(2.55)
    assert np.isnan(df.iloc[0]["odds_h_open"])


def test_espn_scoreboard(fixtures_dir):
    payload = json.loads((fixtures_dir / "espn_chi1.json").read_text())
    df = espn.parse_scoreboard(payload, get_league("CHI1"))
    assert len(df) == 2  # postponed match dropped
    done = df[df["status"] == "FT"].iloc[0]
    assert (done["home"], done["away"], done["hg"], done["ag"]) == ("Universidad de Chile", "Colo-Colo", 2, 1)
    assert done["h_shots"] == 14 and done["a_sot"] == 3 and done["h_possession"] == pytest.approx(55.2)
    assert done["odds_h_close"] == pytest.approx(2.20)  # +120 american
    assert done["season"] == "2025"
    assert done["date"] == pd.Timestamp("2025-03-01")  # 23:30Z is 19:30 local
    sched = df[df["status"] == "SCHEDULED"].iloc[0]
    assert np.isnan(sched["hg"])


def test_american_to_decimal():
    assert espn.american_to_decimal(150) == pytest.approx(2.5)
    assert espn.american_to_decimal(-200) == pytest.approx(1.5)
    assert np.isnan(espn.american_to_decimal("n/a"))


def test_odds_api_parse_and_consensus(fixtures_dir):
    payload = json.loads((fixtures_dir / "odds_api_chile.json").read_text())
    lg = get_league("CHI1")
    s1 = odds_api.parse_odds(payload, lg, taken_at=pd.Timestamp("2025-03-05T10:00Z"))
    assert len(s1) == 2
    assert s1.loc[s1["bookmaker"] == "pinnacle", "odds_o25"].iloc[0] == pytest.approx(2.0)
    later = [dict(ev) for ev in payload]
    later[0] = json.loads(json.dumps(payload[0]))
    later[0]["bookmakers"][0]["markets"][0]["outcomes"][0]["price"] = 1.95
    s2 = odds_api.parse_odds(later, lg, taken_at=pd.Timestamp("2025-03-08T18:00Z"))
    cons = odds_api.consensus(pd.concat([s1, s2]))
    row = cons.iloc[0]
    assert row["odds_h_open"] == pytest.approx((2.10 + 2.20) / 2)
    assert row["odds_h_close"] == pytest.approx((1.95 + 2.20) / 2)
    assert row["odds_h_best"] == pytest.approx(2.20)


def test_international(fixtures_dir):
    df = international.parse((fixtures_dir / "international.csv").read_text(), get_league("INT"))
    assert len(df) == 3
    assert df["neutral"].tolist() == [True, False, True]
    imp = international.tournament_importance(df["tournament"])
    assert imp.tolist() == [3.0, 0.0, 2.0]


def test_team_matching():
    cands = ["Universidad de Chile", "Colo-Colo", "Everton", "O'Higgins", "Manchester United"]
    assert teams.match_name("U. de Chile", cands) == "Universidad de Chile"
    assert teams.match_name("Colo Colo", cands) == "Colo-Colo"
    assert teams.match_name("Everton de Viña del Mar", cands) == "Everton"
    assert teams.match_name("OHiggins", cands) == "O'Higgins"
    assert teams.match_name("Man United", cands) == "Manchester United"
    assert teams.match_name("Real Madrid", cands) is None


def test_football_data_fixtures(fixtures_dir):
    df = football_data.parse_main((fixtures_dir / "football_data_fixtures.csv").read_text(), get_league("EPL"))
    assert len(df) == 1  # only the EPL row
    r = df.iloc[0]
    assert r["home"] == "Arsenal" and np.isnan(r["hg"]) and r["status"] == "SCHEDULED"
    assert r["odds_h_open"] == pytest.approx(2.02) and r["season"] == "2026-27"

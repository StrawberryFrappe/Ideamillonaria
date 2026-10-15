import numpy as np
import pandas as pd
import pytest

from ideamillonaria import journal, pipeline
from ideamillonaria.config import get_league
from ideamillonaria.evaluate import betting, metrics


def test_edge_kelly_stake():
    assert betting.edge(0.5, 2.2) == pytest.approx(0.10)
    assert betting.kelly(0.5, 2.2) == pytest.approx(0.1 / 1.2)
    assert betting.kelly(0.4, 2.0) == 0.0
    # quarter Kelly = 2.08% -> capped at 2% of 1000
    assert betting.suggested_stake(0.5, 2.2, 1000) == pytest.approx(20.0)
    assert betting.suggested_stake(0.5, 2.05, 1000) == pytest.approx(1000 * 0.25 * 0.025 / 1.05, abs=0.01)


def test_simulate_hand_computed():
    bt = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3),
        "y": [0, 2, 1],
        "p_v_h": [0.6, 0.6, 0.2], "p_v_d": [0.2, 0.2, 0.3], "p_v_a": [0.2, 0.2, 0.5],
        "odds_h_open": [2.0, 2.0, 5.0], "odds_d_open": [3.5, 3.5, 3.0], "odds_a_open": [4.0, 4.0, 1.8],
        "odds_h_close": [1.8, 2.1, 5.0], "odds_d_close": [3.5, 3.5, 3.0], "odds_a_close": [4.5, 4.0, 1.8],
    })
    r = betting.simulate(bt, "v", min_edge=0.05)
    # match 1: home edge .2 -> won (+1); match 2: home edge .2 -> lost (-1); match 3: no edge >= 5%
    assert r["bets"] == 2
    assert r["profit_units"] == pytest.approx(0.0)
    assert r["hit_rate"] == pytest.approx(0.5)


def test_rps_known_values():
    y = np.array([0])
    assert metrics.rps(np.array([[1.0, 0, 0]]), y)[0] == 0
    assert metrics.rps(np.array([[0, 0, 1.0]]), y)[0] == pytest.approx(1.0)
    assert metrics.rps(np.array([[0, 1.0, 0]]), y)[0] == pytest.approx(0.5)


def test_full_pipeline_on_synthetic(tmp_data, synthetic):
    df, _ = synthetic
    league = get_league("SYN")
    art, prep = pipeline.train(league, df, verbose=False)
    tr = pd.DataFrame(art.track_record["common"]).set_index("view")
    base = metrics.score(metrics.probs_of(pd.read_parquet(pipeline.outputs("SYN")["backtest"]), "base"),
                         pd.read_parquet(pipeline.outputs("SYN")["backtest"])["y"].to_numpy(int))
    assert tr.loc["ens", "rps"] < base["rps"]  # beats "know nothing"
    assert "betting" in art.track_record
    preds = pipeline.predict(league, art, prep, verbose=False)
    assert len(preds) > 0
    assert np.allclose(preds[["p_ens_h", "p_ens_d", "p_ens_a"]].sum(axis=1), 1)
    assert set(preds["agreement"]) <= {"high", "moderate", "low"}
    loaded = pipeline.load_artifacts("SYN")
    p, X = pipeline.predict_custom(league, loaded, df, preds["home"].iloc[0], preds["away"].iloc[0])
    assert len(p) == 1 and np.isclose(p[["p_ens_h", "p_ens_d", "p_ens_a"]].sum(axis=1).iloc[0], 1)
    # explanations
    from ideamillonaria.report import explain
    contrib = loaded.gbm.contributions(X)[0]
    rs = explain.reasons(contrib, loaded.gbm.features, X.iloc[0], 0, "H", "A")
    assert len(rs) > 0
    assert explain.summary(p.iloc[0], "WWDLW", "LLDWW")


def test_journal_roundtrip(tmp_data):
    m = pd.DataFrame({"date": [pd.Timestamp("2025-03-01")], "home": ["A"], "away": ["B"], "hg": [2.0],
                      "ag": [1.0], "odds_h_close": [1.9]})
    journal.add_bet("CHI1", "2025-03-01", "A", "B", "1X2", "h", 2.0, 10, "ens", 0.55)
    journal.add_bet("CHI1", "2025-03-01", "A", "B", "OU25", "under", 1.8, 5, "dc", 0.6)
    assert journal.settle(m, "CHI1") == 2
    b = journal.load()
    assert b["profit"].tolist() == [10.0, -5.0]
    s = journal.summary(b)
    assert s.loc["ALL", "profit"] == pytest.approx(5.0)
    assert s.loc["ens", "avg_clv"] == pytest.approx(2.0 / 1.9 - 1)

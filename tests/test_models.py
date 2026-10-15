import numpy as np
import pandas as pd
import pytest

from ideamillonaria.config import EloParams
from ideamillonaria.data.schema import conform
from ideamillonaria.models import dixon_coles as dc
from ideamillonaria.models import elo
from ideamillonaria.models.ensemble import blend, fit_weights
from ideamillonaria.models.market import remove_vig
from ideamillonaria.report import markets


def test_score_matrix_normalised_and_outcomes():
    p = dc.score_matrix([1.6, 0.4, 3.0], [1.1, 2.2, 0.3], -0.08)
    assert np.allclose(p.sum(axis=(1, 2)), 1)
    probs = dc.outcome_probs(p)
    assert np.allclose(probs.sum(1), 1)
    assert probs[0, 0] > probs[0, 2] and probs[1, 2] > probs[1, 0]


def test_dixon_coles_recovers_structure(synthetic):
    df, truth = synthetic
    played = df[df["hg"].notna()]
    fit = dc.fit(played, played["date"].max() + pd.Timedelta(days=1), xi=0.0019, ridge=1.0)
    assert 0.1 < fit.home < 0.4  # true home advantage 0.25
    true_att = np.array([truth["att"][t] for t in fit.teams])
    assert np.corrcoef(fit.att, true_att)[0, 1] > 0.5


def test_dixon_coles_gradient(synthetic, monkeypatch):
    import scipy.optimize as so
    df, _ = synthetic
    played = df[df["hg"].notna()].head(300)
    captured = {}
    real = so.minimize

    def spy(f, x0, **kw):
        captured["f"] = f
        return real(f, x0, **kw)

    monkeypatch.setattr(dc, "minimize", spy)
    dc.fit(played, played["date"].max() + pd.Timedelta(days=1), 0.002, 1.0)
    f = captured["f"]
    x = np.random.default_rng(0).normal(0, 0.2, 0)
    n_params = len(set(played["home"]) | set(played["away"])) * 2 + 3
    x = np.random.default_rng(0).normal(0, 0.2, n_params)
    x[-2] = 0.05
    _, g = f(x)
    eps = 1e-6
    num = np.array([(f(x + eps * e)[0] - f(x - eps * e)[0]) / (2 * eps) for e in np.eye(n_params)])
    assert np.allclose(num, g, atol=1e-4)


def test_elo_zero_sum_and_prematch():
    m = conform(pd.DataFrame({
        "league": "T", "season": "2020", "date": pd.date_range("2020-01-01", periods=4, freq="7D"),
        "home": ["A", "B", "A", "C"], "away": ["B", "C", "C", "A"], "hg": [2, 1, 0, np.nan], "ag": [0, 1, 3, np.nan],
    }))
    out, state = elo.compute(m, EloParams(newcomer_offset=0.0))
    assert out.iloc[0]["elo_h"] == out.iloc[0]["elo_a"] == 1500  # first match: no information yet
    assert sum(state.ratings.values()) == pytest.approx(1500 * 3)  # zero-sum
    assert out.iloc[1]["elo_h"] < 1500  # B lost its first match
    assert np.isfinite(out.iloc[3]["elo_diff"])  # scheduled match still gets pre-match ratings


def test_remove_vig():
    p = remove_vig(np.array([[1.9, 3.5, 4.2], [1.25, 6.0, 12.0]]))
    assert np.allclose(p.sum(1), 1)
    prop = remove_vig(np.array([[1.25, 6.0, 12.0]]), "proportional")
    assert p[1, 2] < prop[0, 2]  # power method trims longshots more (favourite-longshot bias)


def test_ensemble_prefers_better_model():
    rng = np.random.default_rng(1)
    true = rng.dirichlet([4, 3, 3], 3000)
    y = np.array([rng.choice(3, p=p) for p in true])
    noisy = rng.dirichlet(np.ones(3), 3000)
    w = fit_weights([true, noisy], y)
    assert w[0] > 0.8
    b = blend([true, np.full_like(true, np.nan)], np.array([0.5, 0.5]))
    assert np.allclose(b, true)  # missing component -> weights renormalise


def test_markets_consistent():
    mk = markets.markets(1.5, 1.1, -0.05, "H", "A")
    get = lambda m, s: mk[(mk["market"] == m) & (mk["selection"] == s)]["probability"].iloc[0]
    assert get("1X2", "H win") + get("1X2", "Draw") + get("1X2", "A win") == pytest.approx(1)
    assert get("Total goals", "Over 2.5") + get("Total goals", "Under 2.5") == pytest.approx(1)
    assert get("Double chance", "H or draw (1X)") == pytest.approx(get("1X2", "H win") + get("1X2", "Draw"))
    dnb = get("Draw no bet", "H")
    assert dnb == pytest.approx(get("1X2", "H win") / (get("1X2", "H win") + get("1X2", "A win")))
    top = markets.top_scorelines(1.5, 1.1, -0.05, 5)
    assert top["probability"].is_monotonic_decreasing

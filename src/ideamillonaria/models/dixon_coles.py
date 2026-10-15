"""Time-weighted Dixon-Coles bivariate Poisson model.

    log lambda_home = mu + home_adv + att[home] - def[away]
    log lambda_away = mu            + att[away] - def[home]

with the Dixon & Coles (1997) low-score correction ``tau(x, y; rho)`` and
exponential time decay ``w = exp(-xi * days_ago)`` on each match's likelihood.
Attack / defence parameters get a Gaussian (ridge) prior, which makes the model
identifiable and shrinks teams with little data towards average - important for
small leagues like Chile.

The output is a full scoreline probability matrix, from which every market
(1X2, over/under, BTTS, handicaps, exact score) is derived in ``report/markets.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


@dataclass
class DCFit:
    teams: list[str]
    att: np.ndarray
    dfn: np.ndarray
    home: float
    rho: float
    mu: float
    ref_date: pd.Timestamp

    @property
    def index(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.teams)}

    def ratings(self) -> pd.DataFrame:
        return pd.DataFrame({"team": self.teams, "attack": self.att, "defence": self.dfn}).sort_values(
            "attack", ascending=False
        )


def _tau_and_grads(x, y, lam, mu_, rho):
    tau = np.ones_like(lam)
    dl = np.zeros_like(lam)  # d log tau / d log lambda
    dm = np.zeros_like(lam)  # d log tau / d log mu
    dr = np.zeros_like(lam)  # d log tau / d rho
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1 - lam[m00] * mu_[m00] * rho
    tau[m01] = 1 + lam[m01] * rho
    tau[m10] = 1 + mu_[m10] * rho
    tau[m11] = 1 - rho
    tau = np.maximum(tau, 1e-10)
    dl[m00] = -lam[m00] * mu_[m00] * rho / tau[m00]
    dm[m00] = dl[m00]
    dr[m00] = -lam[m00] * mu_[m00] / tau[m00]
    dl[m01] = lam[m01] * rho / tau[m01]
    dr[m01] = lam[m01] / tau[m01]
    dm[m10] = mu_[m10] * rho / tau[m10]
    dr[m10] = mu_[m10] / tau[m10]
    dr[m11] = -1.0 / tau[m11]
    return tau, dl, dm, dr


def fit(matches: pd.DataFrame, ref_date, xi: float, ridge: float, init: DCFit | None = None,
        min_weight: float = 1e-3) -> DCFit | None:
    """Fit on played matches strictly before ``ref_date``."""
    ref_date = pd.Timestamp(ref_date)
    m = matches[(matches["date"] < ref_date) & matches["hg"].notna() & matches["ag"].notna()]
    days = (ref_date - m["date"]).dt.days.to_numpy(float)
    w = np.exp(-xi * days)
    keep = w >= min_weight
    m, w = m[keep], w[keep]
    if len(m) < 30:
        return None
    teams = sorted(set(m["home"]) | set(m["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = m["home"].map(idx).to_numpy()
    ai = m["away"].map(idx).to_numpy()
    x = m["hg"].to_numpy(float)
    y = m["ag"].to_numpy(float)
    home_flag = (~m["neutral"].to_numpy(bool)).astype(float)

    theta0 = np.zeros(2 * n + 3)
    theta0[2 * n] = 0.25
    theta0[2 * n + 1] = -0.05
    theta0[2 * n + 2] = np.log(max(np.average((x + y) / 2, weights=w), 0.1))
    if init is not None:
        old = init.index
        for t, i in idx.items():
            if t in old:
                theta0[i] = init.att[old[t]]
                theta0[n + i] = init.dfn[old[t]]
        theta0[2 * n:] = [init.home, init.rho, init.mu]

    def f(theta):
        att, dfn = theta[:n], theta[n:2 * n]
        home, rho, mu = theta[2 * n:]
        loglam = mu + home * home_flag + att[hi] - dfn[ai]
        logmu = mu + att[ai] - dfn[hi]
        lam, mmu = np.exp(loglam), np.exp(logmu)
        tau, dl, dm, dr = _tau_and_grads(x, y, lam, mmu, rho)
        ll = w * (np.log(tau) + x * loglam - lam + y * logmu - mmu)
        gl = w * (x - lam + dl)
        gm = w * (y - mmu + dm)
        g = np.zeros_like(theta)
        g[:n] = np.bincount(hi, gl, n) + np.bincount(ai, gm, n)
        g[n:2 * n] = -np.bincount(ai, gl, n) - np.bincount(hi, gm, n)
        g[2 * n] = np.sum(gl * home_flag)
        g[2 * n + 1] = np.sum(w * dr)
        g[2 * n + 2] = np.sum(gl + gm)
        loss = -ll.sum() + ridge * (np.sum(att ** 2) + np.sum(dfn ** 2))
        grad = -g
        grad[:n] += 2 * ridge * att
        grad[n:2 * n] += 2 * ridge * dfn
        return loss, grad

    bounds = [(None, None)] * (2 * n) + [(-1.0, 1.5), (-0.3, 0.3), (None, None)]
    res = minimize(f, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500, "gtol": 1e-6})
    th = res.x
    return DCFit(teams, th[:n].copy(), th[n:2 * n].copy(), float(th[2 * n]), float(th[2 * n + 1]),
                 float(th[2 * n + 2]), ref_date)


def expected_goals(model: DCFit, home, away, neutral) -> tuple[np.ndarray, np.ndarray]:
    idx = model.index
    # unseen (e.g. promoted) teams get a below-average profile
    new_att = np.quantile(model.att, 0.25) if len(model.att) else 0.0
    new_def = np.quantile(model.dfn, 0.25) if len(model.dfn) else 0.0
    home = np.asarray(home, dtype=object)
    away = np.asarray(away, dtype=object)
    neutral = np.asarray(neutral, dtype=bool)
    ah = np.array([model.att[idx[t]] if t in idx else new_att for t in home])
    dh = np.array([model.dfn[idx[t]] if t in idx else new_def for t in home])
    aa = np.array([model.att[idx[t]] if t in idx else new_att for t in away])
    da = np.array([model.dfn[idx[t]] if t in idx else new_def for t in away])
    lam = np.exp(model.mu + model.home * (~neutral) + ah - da)
    mu_ = np.exp(model.mu + aa - dh)
    return lam, mu_


def score_matrix(lam, mu_, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """P[i, x, y] for each match i (rows = home goals, cols = away goals)."""
    lam = np.atleast_1d(np.asarray(lam, float))
    mu_ = np.atleast_1d(np.asarray(mu_, float))
    g = np.arange(max_goals + 1)
    ph = poisson.pmf(g[None, :], lam[:, None])
    pa = poisson.pmf(g[None, :], mu_[:, None])
    p = ph[:, :, None] * pa[:, None, :]
    p[:, 0, 0] *= np.maximum(1 - lam * mu_ * rho, 0)
    p[:, 0, 1] *= np.maximum(1 + lam * rho, 0)
    p[:, 1, 0] *= np.maximum(1 + mu_ * rho, 0)
    p[:, 1, 1] *= max(1 - rho, 0)
    p /= p.sum(axis=(1, 2), keepdims=True)
    return p


def outcome_probs(p: np.ndarray) -> np.ndarray:
    """(n, 3) home / draw / away probabilities from score matrices."""
    home = np.tril(np.ones(p.shape[1:]), -1).astype(bool)
    away = np.triu(np.ones(p.shape[1:]), 1).astype(bool)
    return np.column_stack([p[:, home].sum(1), np.trace(p, axis1=1, axis2=2), p[:, away].sum(1)])


def predict(model: DCFit, df: pd.DataFrame) -> pd.DataFrame:
    lam, mu_ = expected_goals(model, df["home"], df["away"], df["neutral"])
    p = score_matrix(lam, mu_, model.rho)
    probs = outcome_probs(p)
    goals = np.arange(p.shape[1])
    total = goals[:, None] + goals[None, :]
    return pd.DataFrame({
        "dc_lambda_h": lam,
        "dc_lambda_a": mu_,
        "p_dc_h": probs[:, 0],
        "p_dc_d": probs[:, 1],
        "p_dc_a": probs[:, 2],
        "p_dc_o25": p[:, total > 2].sum(1),
        "p_dc_btts": p[:, 1:, 1:].sum(axis=(1, 2)),
    }, index=df.index)


def rolling_predict(matches: pd.DataFrame, xi: float, ridge: float, refit_days: int, window_days: int,
                    start, verbose: bool = False) -> tuple[pd.DataFrame, DCFit | None]:
    """Walk forward through time: refit every ``refit_days`` on matches strictly before the
    block start and predict the block. Every prediction is out-of-sample."""
    matches = matches.sort_values("date")
    start = pd.Timestamp(start)
    dates = matches["date"]
    out = []
    t0 = max(start, dates.min())
    last_fit: DCFit | None = None
    end = dates.max()
    n_fits = 0
    while t0 <= end:
        upcoming = dates[dates >= t0]
        if upcoming.empty:
            break
        t0 = max(t0, upcoming.iloc[0].normalize())
        t1 = t0 + pd.Timedelta(days=refit_days)
        block = matches[(dates >= t0) & (dates < t1)]
        window = matches[(dates >= t0 - pd.Timedelta(days=window_days)) & (dates < t0)]
        fitted = fit(window, t0, xi, ridge, init=last_fit)
        n_fits += 1
        if fitted is not None:
            last_fit = fitted
            out.append(predict(fitted, block))
        t0 = t1
    if verbose:
        print(f"  Dixon-Coles: {n_fits} walk-forward fits")
    if not out:
        return pd.DataFrame(), last_fit
    return pd.concat(out), last_fit

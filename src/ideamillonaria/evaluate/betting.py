"""Read-only betting layer: edges, fair odds, stake suggestions and backtests.

Nothing in this module feeds back into model training or selection. It takes frozen
model probabilities plus bookmaker odds and answers "if I had followed view X, what
would have happened?" - with confidence intervals, because a few hundred bets are a
very noisy sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ["h", "d", "a"]


def fair_odds(p) -> np.ndarray:
    p = np.asarray(p, float)
    with np.errstate(divide="ignore"):
        return np.where(p > 0, 1.0 / p, np.inf)


def edge(p, odds) -> np.ndarray:
    """Expected profit per unit staked: p * odds - 1."""
    return np.asarray(p, float) * np.asarray(odds, float) - 1.0


def kelly(p, odds) -> np.ndarray:
    p, o = np.asarray(p, float), np.asarray(odds, float)
    return np.clip((p * o - 1.0) / (o - 1.0), 0.0, None)


def suggested_stake(p: float, odds: float, bankroll: float, fraction: float = 0.25, cap: float = 0.02) -> float:
    """Fractional Kelly, capped at ``cap`` of the bankroll. A suggestion, not advice."""
    f = float(kelly(p, odds)) * fraction
    return round(bankroll * min(f, cap), 2)


def pick_bets(probs: np.ndarray, odds: np.ndarray, min_edge: float = 0.05, max_odds: float = 7.0,
              min_prob: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """For each match pick the outcome with the largest positive edge (or -1 for no bet)."""
    e = edge(probs, odds)
    e = np.where((odds <= max_odds) & (probs >= min_prob) & np.isfinite(e), e, -np.inf)
    sel = np.argmax(e, axis=1)
    best = e[np.arange(len(e)), sel]
    sel = np.where(best >= min_edge, sel, -1)
    return sel, best


def simulate(bt: pd.DataFrame, view: str, when: str = "open", min_edge: float = 0.05,
             max_odds: float = 7.0, kelly_fraction: float = 0.25, kelly_cap: float = 0.02,
             n_boot: int = 2000, seed: int = 0) -> dict:
    """Flat-stake and fractional-Kelly backtest of following one view's value bets."""
    cols = [f"p_{view}_{o}" for o in OUTCOMES]
    ocols = [f"odds_{o}_{when}" for o in OUTCOMES]
    need = cols + ocols
    d = bt.dropna(subset=need)
    if d.empty:
        return {"view": view, "bets": 0}
    probs = d[cols].to_numpy(float)
    odds = d[ocols].to_numpy(float)
    y = d["y"].to_numpy(int)
    sel, e = pick_bets(probs, odds, min_edge, max_odds)
    bet = sel >= 0
    if bet.sum() == 0:
        return {"view": view, "bets": 0, "matches": int(len(d))}
    s = sel[bet]
    o = odds[bet, s]
    won = (y[bet] == s)
    profit = np.where(won, o - 1.0, -1.0)
    # bootstrap CI on flat-stake ROI
    rng = np.random.default_rng(seed)
    boots = rng.choice(profit, size=(n_boot, len(profit)), replace=True).mean(axis=1)
    # closing line value: did we beat the (vig-free) closing price?
    clv = np.nan
    ccols = [f"odds_{o}_close" for o in OUTCOMES]
    if when == "open" and all(c in d for c in ccols):
        from ..models.market import remove_vig
        close = d[ccols].to_numpy(float)[bet]
        okc = ~np.isnan(close).any(axis=1)
        if okc.any():
            pc = remove_vig(close[okc])
            fair_close = 1.0 / pc[np.arange(okc.sum()), s[okc]]
            clv = float(np.mean(o[okc] / fair_close - 1.0))
    # fractional Kelly bankroll path (bets settled in date order)
    bank = 1.0
    path = []
    p_sel = probs[bet, s]
    for pi, oi, wi in zip(p_sel, o, won):
        stake = bank * min(float(kelly(pi, oi)) * kelly_fraction, kelly_cap)
        bank += stake * (oi - 1.0) if wi else -stake
        path.append(bank)
    path = np.array(path)
    peak = np.maximum.accumulate(np.concatenate([[1.0], path]))[1:]
    flat_curve = np.cumsum(profit)
    return {
        "view": view,
        "matches": int(len(d)),
        "bets": int(bet.sum()),
        "hit_rate": float(won.mean()),
        "avg_odds": float(o.mean()),
        "avg_edge": float(e[bet].mean()),
        "profit_units": float(profit.sum()),
        "roi": float(profit.mean()),
        "roi_ci_low": float(np.quantile(boots, 0.025)),
        "roi_ci_high": float(np.quantile(boots, 0.975)),
        "p_roi_positive": float((boots > 0).mean()),
        "clv": clv,
        "kelly_final_bankroll": float(path[-1]),
        "kelly_max_drawdown": float(np.max(1 - path / peak)),
        "_curve_dates": d["date"].to_numpy()[bet],
        "_curve_flat": flat_curve,
    }


def simulate_all(bt: pd.DataFrame, views: list[str], **kw) -> pd.DataFrame:
    rows = [simulate(bt, v, **kw) for v in views]
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    return df.set_index("view") if not df.empty else df


def equity_curves(bt: pd.DataFrame, views: list[str], **kw) -> pd.DataFrame:
    frames = []
    for v in views:
        r = simulate(bt, v, **kw)
        if r.get("bets", 0) > 0:
            frames.append(pd.DataFrame({"date": r["_curve_dates"], "profit_units": r["_curve_flat"],
                                        "bet_no": np.arange(1, len(r["_curve_flat"]) + 1), "view": v}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

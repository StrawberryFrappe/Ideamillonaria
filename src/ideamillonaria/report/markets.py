"""Every common betting market derived from one Dixon-Coles scoreline matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.dixon_coles import score_matrix


def _fair(p_win: float, p_lose: float) -> float:
    """Break-even decimal odds for a bet that can also push (stake returned)."""
    if p_win <= 0:
        return np.inf
    return 1.0 + p_lose / p_win


def markets(lam: float, mu: float, rho: float, home: str = "Home", away: str = "Away") -> pd.DataFrame:
    p = score_matrix([lam], [mu], rho)[0]
    g = np.arange(p.shape[0])
    margin = g[:, None] - g[None, :]
    total = g[:, None] + g[None, :]
    rows = []

    def add(market, selection, win, push=0.0):
        lose = max(1.0 - win - push, 0.0)
        rows.append({"market": market, "selection": selection,
                     "probability": win / (win + lose) if push else win,
                     "fair_odds": _fair(win, lose)})

    ph, pd_, pa = p[margin > 0].sum(), p[margin == 0].sum(), p[margin < 0].sum()
    add("1X2", f"{home} win", ph)
    add("1X2", "Draw", pd_)
    add("1X2", f"{away} win", pa)
    add("Double chance", f"{home} or draw (1X)", ph + pd_)
    add("Double chance", f"{home} or {away} (12)", ph + pa)
    add("Double chance", f"Draw or {away} (X2)", pd_ + pa)
    add("Draw no bet", home, ph, pd_)
    add("Draw no bet", away, pa, pd_)
    for line in (0.5, 1.5, 2.5, 3.5, 4.5):
        over = p[total > line].sum()
        add("Total goals", f"Over {line}", over)
        add("Total goals", f"Under {line}", 1 - over)
    btts = p[1:, 1:].sum()
    add("Both teams to score", "Yes", btts)
    add("Both teams to score", "No", 1 - btts)
    # Asian handicaps (home perspective)
    add("Asian handicap", f"{home} -0.5", ph)
    add("Asian handicap", f"{away} +0.5", pd_ + pa)
    add("Asian handicap", f"{home} -1", p[margin >= 2].sum(), p[margin == 1].sum())
    add("Asian handicap", f"{away} +1", p[margin <= 0].sum(), p[margin == 1].sum())
    add("Asian handicap", f"{home} -1.5", p[margin >= 2].sum())
    add("Asian handicap", f"{away} +1.5", p[margin <= 1].sum())
    add("Asian handicap", f"{home} +0.5", ph + pd_)
    add("Asian handicap", f"{away} -0.5", pa)
    # team totals & clean sheets
    hg_marg, ag_marg = p.sum(1), p.sum(0)
    add("Team goals", f"{home} over 0.5", 1 - hg_marg[0])
    add("Team goals", f"{home} over 1.5", hg_marg[2:].sum())
    add("Team goals", f"{away} over 0.5", 1 - ag_marg[0])
    add("Team goals", f"{away} over 1.5", ag_marg[2:].sum())
    add("Clean sheet", home, ag_marg[0])
    add("Clean sheet", away, hg_marg[0])
    return pd.DataFrame(rows)


def top_scorelines(lam: float, mu: float, rho: float, n: int = 10) -> pd.DataFrame:
    p = score_matrix([lam], [mu], rho)[0]
    flat = [(i, j, p[i, j]) for i in range(p.shape[0]) for j in range(p.shape[1])]
    flat.sort(key=lambda t: -t[2])
    return pd.DataFrame([{"score": f"{i}-{j}", "probability": q, "fair_odds": 1 / q} for i, j, q in flat[:n]])


def matrix(lam: float, mu: float, rho: float, max_goals: int = 6) -> np.ndarray:
    return score_matrix([lam], [mu], rho)[0][: max_goals + 1, : max_goals + 1]

"""Bookmaker odds -> fair probabilities, and the market feature set.

Bookmakers build a margin ("vig" / overround) into their prices, so 1/odds sums to more
than 1. We strip it with the *power* method (solve sum((1/o)^k) = 1), which handles the
favourite-longshot bias better than simple normalisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def overround(*odds) -> np.ndarray:
    return sum(1.0 / np.asarray(o, float) for o in odds) - 1.0


def remove_vig(odds: np.ndarray, method: str = "power", iters: int = 60) -> np.ndarray:
    """``odds``: (n, k) decimal odds -> (n, k) fair probabilities (rows with NaN stay NaN)."""
    odds = np.atleast_2d(np.asarray(odds, float))
    inv = 1.0 / odds
    if method == "proportional":
        return inv / inv.sum(axis=1, keepdims=True)
    # power method via bisection on k in [0.5, 3]
    lo = np.full(len(odds), 0.5)
    hi = np.full(len(odds), 3.0)
    for _ in range(iters):
        k = (lo + hi) / 2
        s = np.sum(inv ** k[:, None], axis=1)
        too_big = s > 1  # need larger k to shrink the sum
        lo = np.where(too_big, k, lo)
        hi = np.where(too_big, hi, k)
    k = (lo + hi) / 2
    p = inv ** k[:, None]
    return p / p.sum(axis=1, keepdims=True)


def book_probs(df: pd.DataFrame, when: str = "open") -> pd.DataFrame:
    """Fair 1X2 and over-2.5 probabilities implied by the ``open`` or ``close`` prices."""
    p = remove_vig(df[[f"odds_h_{when}", f"odds_d_{when}", f"odds_a_{when}"]].to_numpy())
    ou = remove_vig(df[[f"odds_o25_{when}", f"odds_u25_{when}"]].to_numpy())
    return pd.DataFrame({
        f"p_book_{when}_h": p[:, 0],
        f"p_book_{when}_d": p[:, 1],
        f"p_book_{when}_a": p[:, 2],
        f"p_book_{when}_o25": ou[:, 0],
        f"book_margin_{when}": overround(df[f"odds_h_{when}"], df[f"odds_d_{when}"], df[f"odds_a_{when}"]),
    }, index=df.index)


MARKET_FEATURES = ["p_book_open_h", "p_book_open_d", "p_book_open_a", "p_book_open_o25", "book_margin_open"]

"""Proper scoring rules for H/D/A forecasts. These (never betting ROI) are what we
use to compare and select models.

* RPS (ranked probability score): the standard football metric; respects the order
  home < draw < away, so predicting a draw when the home side wins is "less wrong"
  than predicting an away win. Lower is better (~0.19-0.21 is typical for top leagues).
* Log-loss: punishes confident mistakes hard. Lower is better (uniform guess = 1.0986).
* Brier: mean squared error of the probability vector. Lower is better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VIEWS = {
    "elo": "Elo rating",
    "dc": "Dixon-Coles goals model",
    "gbm": "Stats model (pure, no odds)",
    "mkt": "Market-aware model (stats + opening odds)",
    "ens": "Ensemble (Elo + DC + stats)",
    "book_open": "Bookmakers, opening (vig removed)",
    "book_close": "Bookmakers, closing (vig removed)",
    "base": "Base rates (always the league average)",
}


def probs_of(df: pd.DataFrame, view: str) -> np.ndarray:
    return df[[f"p_{view}_h", f"p_{view}_d", f"p_{view}_a"]].to_numpy(float)


def rps(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    o = np.zeros_like(p)
    o[np.arange(len(y)), y] = 1
    cp, co = np.cumsum(p, 1)[:, :2], np.cumsum(o, 1)[:, :2]
    return np.sum((cp - co) ** 2, axis=1) / 2


def logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    o = np.zeros_like(p)
    o[np.arange(len(y)), y] = 1
    return np.sum((p - o) ** 2, axis=1)


def score(p: np.ndarray, y: np.ndarray) -> dict:
    ok = ~np.isnan(p).any(axis=1)
    p, y = p[ok], y[ok]
    if len(y) == 0:
        return {"n": 0, "rps": np.nan, "logloss": np.nan, "brier": np.nan, "accuracy": np.nan}
    return {
        "n": int(len(y)),
        "rps": float(rps(p, y).mean()),
        "logloss": float(logloss(p, y).mean()),
        "brier": float(brier(p, y).mean()),
        "accuracy": float((p.argmax(1) == y).mean()),
    }


def metrics_table(bt: pd.DataFrame, views: list[str], common: bool = False) -> pd.DataFrame:
    """One row per view. ``common=True`` scores all views on the same matches only
    (rows where every listed view has a prediction) for an apples-to-apples comparison."""
    y = bt["y"].to_numpy(int)
    mask = np.ones(len(bt), bool)
    if common:
        for v in views:
            mask &= ~np.isnan(probs_of(bt, v)).any(axis=1)
    rows = []
    for v in views:
        if f"p_{v}_h" not in bt:
            continue
        s = score(probs_of(bt, v)[mask], y[mask])
        rows.append({"view": v, "description": VIEWS.get(v, v), **s})
    return pd.DataFrame(rows).set_index("view")


def calibration(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Reliability table: predicted probability vs observed frequency, all outcomes pooled."""
    ok = ~np.isnan(p).any(axis=1)
    p, y = p[ok], y[ok]
    o = np.zeros_like(p)
    o[np.arange(len(y)), y] = 1
    flat_p, flat_o = p.ravel(), o.ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(flat_p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "p": flat_p, "o": flat_o})
    g = df.groupby("bin").agg(predicted=("p", "mean"), observed=("o", "mean"), n=("o", "size"))
    return g.reset_index(drop=True)

"""Blend several probability forecasts with weights fit on out-of-sample log-loss."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _blend(probs: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    stack = np.stack(probs)  # (k, n, 3)
    avail = ~np.isnan(stack).any(axis=2)  # (k, n)
    ww = w[:, None] * avail
    ww = ww / np.where(ww.sum(0) > 0, ww.sum(0), 1)
    out = np.nansum(stack * ww[:, :, None], axis=0)
    out[~avail.any(0)] = np.nan
    return out


def fit_weights(probs: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    stack = np.stack(probs)
    ok = ~np.isnan(stack).any(axis=(0, 2))
    if ok.sum() < 50:
        return np.full(len(probs), 1.0 / len(probs))
    s = stack[:, ok]
    yy = y[ok]

    def loss(z):
        w = np.exp(z) / np.exp(z).sum()
        p = np.tensordot(w, s, axes=1)
        return -np.mean(np.log(np.clip(p[np.arange(len(yy)), yy], 1e-12, 1)))

    res = minimize(loss, np.zeros(len(probs)), method="Nelder-Mead", options={"maxiter": 2000})
    w = np.exp(res.x)
    return w / w.sum()


def blend(probs: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return _blend(probs, np.asarray(weights, float))

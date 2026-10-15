"""LightGBM multiclass (home / draw / away) model with temperature calibration.

Runs on CPU in seconds for this data size. Set IDEAM_GBM_DEVICE=gpu to try the GPU
(needs a GPU-enabled LightGBM build); it falls back to CPU automatically.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .. import config

PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbose": -1,
    "seed": 42,
    "num_threads": 0,
}


def _softmax_temp(p: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-12, 1)) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))


class GBMModel:
    def __init__(self, features: list[str], params: dict | None = None):
        self.features = list(features)
        self.params = {**PARAMS, **(params or {})}
        if config.GBM_DEVICE == "gpu":
            self.params["device_type"] = "gpu"
        self.booster: lgb.Booster | None = None
        self.temperature = 1.0
        self.best_iteration = 300

    def _train(self, dtrain, rounds, valid=None, callbacks=None):
        try:
            return lgb.train(self.params, dtrain, num_boost_round=rounds,
                             valid_sets=[valid] if valid is not None else None, callbacks=callbacks)
        except lgb.basic.LightGBMError:
            if self.params.get("device_type") == "gpu":
                warnings.warn("LightGBM GPU not available, falling back to CPU")
                self.params.pop("device_type")
                return self._train(dtrain, rounds, valid, callbacks)
            raise

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame | None = None,
            y_val: np.ndarray | None = None, rounds: int | None = None) -> "GBMModel":
        dtrain = lgb.Dataset(X[self.features], label=y, free_raw_data=False)
        if X_val is not None and len(X_val) > 50:
            dval = lgb.Dataset(X_val[self.features], label=y_val, reference=dtrain)
            self.booster = self._train(dtrain, 2000, dval,
                                       [lgb.early_stopping(100, verbose=False)])
            self.best_iteration = max(int(self.booster.best_iteration or 300), 20)
            p_val = self.booster.predict(X_val[self.features], num_iteration=self.best_iteration)
            res = minimize_scalar(lambda t: _logloss(_softmax_temp(p_val, t), y_val), bounds=(0.5, 3.0),
                                  method="bounded")
            self.temperature = float(res.x)
        else:
            self.best_iteration = rounds or self.best_iteration
            self.booster = self._train(dtrain, self.best_iteration)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.booster.predict(X[self.features], num_iteration=self.best_iteration)
        return _softmax_temp(p, self.temperature)

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values, shape (n, 3 classes, n_features + 1 bias) in raw log-odds space."""
        c = self.booster.predict(X[self.features], num_iteration=self.best_iteration, pred_contrib=True)
        k = len(self.features) + 1
        return np.stack([c[:, i * k:(i + 1) * k] for i in range(3)], axis=1)

    def importance(self) -> pd.Series:
        return pd.Series(self.booster.feature_importance("gain"), index=self.features).sort_values(ascending=False)

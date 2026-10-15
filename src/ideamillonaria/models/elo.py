"""Elo ratings for football (eloratings.net style).

* Home advantage added to the home rating (not on neutral ground).
* Goal-difference multiplier: 1, 1.5, then (11 + gd) / 8.
* Between seasons ratings regress towards the mean; new / promoted teams enter
  below the mean (``newcomer_offset``).
* For international matches the K-factor scales with tournament importance.

``compute`` walks the matches in date order and records each team's rating *before*
the match, so the output can be used as a leak-free feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..config import EloParams
from ..data.schema import outcome

MEAN = 1500.0


def expected_home(r_home: float, r_away: float, hfa: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(r_home + hfa - r_away) / 400.0))


def gd_multiplier(gd: int) -> float:
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


@dataclass
class EloState:
    ratings: dict[str, float] = field(default_factory=dict)
    last_season: dict[str, str] = field(default_factory=dict)


def compute(matches: pd.DataFrame, params: EloParams, importance: pd.Series | None = None,
            state: EloState | None = None) -> tuple[pd.DataFrame, EloState]:
    """Return pre-match ratings for every row (played or scheduled) and the final state."""
    state = state or EloState()
    ratings = state.ratings
    last_season = state.last_season
    known_mean = MEAN
    n = len(matches)
    pre_h = np.empty(n)
    pre_a = np.empty(n)
    hfa_arr = np.where(matches["neutral"].to_numpy(bool), 0.0, params.home_adv)
    hg = matches["hg"].to_numpy(float)
    ag = matches["ag"].to_numpy(float)
    seasons = matches["season"].to_numpy(object)
    homes = matches["home"].to_numpy(object)
    aways = matches["away"].to_numpy(object)
    imp = importance.to_numpy(float) if importance is not None else None
    dates = matches["date"].to_numpy()
    pending: list[tuple[str, str, float]] = []  # same-day updates applied once the day is over

    for i in range(n):
        if pending and dates[i] != dates[i - 1]:
            for ph, pa, d in pending:
                ratings[ph] += d
                ratings[pa] -= d
            pending.clear()
        h, a, s = homes[i], aways[i], seasons[i]
        for t in (h, a):
            if t not in ratings:
                # newcomers enter below the current mean of the league
                known_mean = np.mean(list(ratings.values())) if ratings else MEAN
                ratings[t] = known_mean + (params.newcomer_offset if ratings else 0.0)
                last_season[t] = s
            elif last_season[t] != s:
                if params.season_regress > 0:
                    m = np.mean(list(ratings.values()))
                    ratings[t] = m + (ratings[t] - m) * (1 - params.season_regress)
                last_season[t] = s
        rh, ra = ratings[h], ratings[a]
        pre_h[i], pre_a[i] = rh, ra
        if np.isnan(hg[i]) or np.isnan(ag[i]):
            continue
        e = expected_home(rh, ra, hfa_arr[i])
        res = 1.0 if hg[i] > ag[i] else (0.5 if hg[i] == ag[i] else 0.0)
        k = params.k
        if imp is not None:
            k = params.k * (0.5 + 0.5 * imp[i])  # friendlies 0.5x ... World Cup 2x
        pending.append((h, a, k * gd_multiplier(int(hg[i] - ag[i])) * (res - e)))
    for ph, pa, d in pending:
        ratings[ph] += d
        ratings[pa] -= d

    out = pd.DataFrame(
        {"elo_h": pre_h, "elo_a": pre_a, "elo_diff": pre_h + hfa_arr - pre_a},
        index=matches.index,
    )
    return out, state


class EloOutcomeModel:
    """Maps the (home-advantage adjusted) Elo difference to H/D/A probabilities
    with a multinomial logistic regression fitted on past matches."""

    def __init__(self):
        self.clf = LogisticRegression(C=1.0, max_iter=1000)

    @staticmethod
    def _x(elo_diff) -> np.ndarray:
        d = np.asarray(elo_diff, float) / 100.0
        return np.column_stack([d, np.abs(d)])  # |d| lets the draw rate shrink for mismatches

    def fit(self, elo_diff, hg, ag) -> "EloOutcomeModel":
        self.clf.fit(self._x(elo_diff), outcome(hg, ag))
        return self

    def predict_proba(self, elo_diff) -> np.ndarray:
        p = self.clf.predict_proba(self._x(elo_diff))
        out = np.zeros((len(p), 3))
        out[:, self.clf.classes_] = p
        return out

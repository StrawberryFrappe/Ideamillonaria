"""Pre-match feature engineering.

Every feature for a match uses only information from strictly earlier matches
(``shift(1)`` before every rolling window). ``tests/test_leakage.py`` checks this by
deleting all later matches and verifying the features don't change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .market import MARKET_FEATURES, book_probs

WINDOWS = (5, 10)
_ROLL_COLS = ["pts", "gf", "ga", "shots_f", "shots_a", "sot_f", "sot_a", "surprise"]


def _team_long(m: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match (one per team), in the team's perspective."""
    played = m["hg"].notna()
    exp_home = 1.0 / (1.0 + 10 ** (-m["elo_diff"] / 400.0)) if "elo_diff" in m else 0.5
    res_home = np.where(m["hg"] > m["ag"], 1.0, np.where(m["hg"] == m["ag"], 0.5, 0.0))
    res_home = np.where(played, res_home, np.nan)

    def side(is_home: bool) -> pd.DataFrame:
        p, o = ("h", "a") if is_home else ("a", "h")
        gf = m["hg"] if is_home else m["ag"]
        ga = m["ag"] if is_home else m["hg"]
        pts = np.where(gf > ga, 3.0, np.where(gf == ga, 1.0, 0.0))
        res = res_home if is_home else 1.0 - res_home
        exp = exp_home if is_home else 1.0 - exp_home
        return pd.DataFrame({
            "row": m.index,
            "date": m["date"],
            "season": m["season"],
            "team": m["home"] if is_home else m["away"],
            "is_home": is_home and True,
            "gf": gf, "ga": ga,
            "pts": np.where(played, pts, np.nan),
            "shots_f": m[f"{p}_shots"], "shots_a": m[f"{o}_shots"],
            "sot_f": m[f"{p}_sot"], "sot_a": m[f"{o}_sot"],
            "surprise": res - exp,  # results above/below what ratings expected
        })

    long = pd.concat([side(True), side(False)], ignore_index=True)
    return long.sort_values(["team", "date", "row"], kind="stable").reset_index(drop=True)


def _rolling(long: pd.DataFrame, keys: list[str], cols: list[str], window: int) -> pd.DataFrame:
    shifted = long.groupby(keys, sort=False)[cols].shift(1)
    for k in keys:
        shifted[k] = long[k]
    r = shifted.groupby(keys, sort=False)[cols].rolling(window, min_periods=1).mean()
    return r.reset_index(level=list(range(len(keys))), drop=True).sort_index()


def team_form(m: pd.DataFrame) -> pd.DataFrame:
    long = _team_long(m)
    feats = pd.DataFrame(index=long.index)
    for w in WINDOWS:
        r = _rolling(long, ["team"], _ROLL_COLS, w)
        for c in _ROLL_COLS:
            feats[f"form_{c}_{w}"] = r[c]
    rv = _rolling(long, ["team", "is_home"], ["pts", "gf", "ga"], 5)
    for c in ["pts", "gf", "ga"]:
        feats[f"venue_{c}_5"] = rv[c]
    # season to date
    gs = long.groupby(["team", "season"], sort=False)
    prev_pts = gs["pts"].shift(1)
    prev_gd = (long["gf"] - long["ga"]).groupby([long["team"], long["season"]], sort=False).shift(1)
    keys = [long["team"], long["season"]]
    for name, s in (("season_ppg", prev_pts), ("season_gd_pg", prev_gd)):
        total = s.fillna(0.0).groupby(keys, sort=False).cumsum()
        count = s.notna().astype(float).groupby(keys, sort=False).cumsum()
        feats[name] = (total / count).where(count > 0)
    feats["season_played"] = gs.cumcount()
    prev_date = long.groupby("team", sort=False)["date"].shift(1)
    feats["rest_days"] = (long["date"] - prev_date).dt.days.clip(upper=30)
    feats["matches_played"] = long.groupby("team", sort=False).cumcount()
    feats["row"] = long["row"]
    feats["is_home"] = long["is_home"]
    return feats


def head_to_head(m: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    first_is_home = m["home"] <= m["away"]
    pair = np.where(first_is_home, m["home"] + "|" + m["away"], m["away"] + "|" + m["home"])
    gd_first = np.where(first_is_home, m["hg"] - m["ag"], m["ag"] - m["hg"])
    tmp = pd.DataFrame({"pair": pair, "gd": gd_first, "date": m["date"]}, index=m.index)
    tmp = tmp.sort_values(["pair", "date"], kind="stable")
    tmp["prev"] = tmp.groupby("pair")["gd"].shift(1)
    tmp["h2h_mean"] = (tmp.groupby("pair", sort=False)["prev"].rolling(window, min_periods=1).mean()
                       .reset_index(level=0, drop=True))
    tmp["h2h_n"] = tmp["prev"].notna().astype(float).groupby(tmp["pair"], sort=False).cumsum()
    tmp = tmp.loc[m.index]
    sign = np.where(first_is_home, 1.0, -1.0)
    return pd.DataFrame({"h2h_gd_5": (tmp["h2h_mean"] * sign).fillna(0.0), "h2h_n": tmp["h2h_n"].fillna(0)},
                        index=m.index)


def build(matches: pd.DataFrame, elo: pd.DataFrame, dc: pd.DataFrame | None,
          importance: pd.Series | None = None) -> pd.DataFrame:
    """Return a feature frame aligned with ``matches.index``."""
    m = matches.join(elo)
    tf = team_form(m)
    home = tf[tf["is_home"]].set_index("row").drop(columns="is_home").add_prefix("h_")
    away = tf[~tf["is_home"]].set_index("row").drop(columns="is_home").add_prefix("a_")
    X = pd.concat([elo, home, away], axis=1).loc[matches.index]
    X = X.join(head_to_head(matches))
    for c in ["form_pts_5", "form_pts_10", "form_gf_10", "form_ga_10", "form_surprise_10", "season_ppg",
              "form_sot_f_5"]:
        X[f"diff_{c}"] = X[f"h_{c}"] - X[f"a_{c}"]
    X["neutral"] = matches["neutral"].astype(float)
    X["importance"] = importance.loc[matches.index].astype(float) if importance is not None else 1.0
    if dc is not None and not dc.empty:
        X = X.join(dc[["dc_lambda_h", "dc_lambda_a", "p_dc_h", "p_dc_d", "p_dc_a", "p_dc_o25"]])
    else:
        for c in ["dc_lambda_h", "dc_lambda_a", "p_dc_h", "p_dc_d", "p_dc_a", "p_dc_o25"]:
            X[c] = np.nan
    X = X.join(book_probs(matches, "open"))
    return X


def pure_feature_names(X: pd.DataFrame) -> list[str]:
    from ..data.schema import MARKET_MARKERS
    return [c for c in X.columns if not any(mk in c for mk in MARKET_MARKERS)]


def market_feature_names(X: pd.DataFrame) -> list[str]:
    return pure_feature_names(X) + MARKET_FEATURES

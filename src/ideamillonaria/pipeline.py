"""Train -> backtest -> final fit -> predict, and the saved artifacts the dashboard reads."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from . import config
from .config import League, get_league
from .data import store
from .data.schema import conform, outcome
from .evaluate import backtest as bt_mod
from .evaluate import betting, metrics
from .models import dixon_coles, elo, ensemble
from .models import features as feats
from .models.gbm import GBMModel
from .models.market import MARKET_FEATURES

BET_VIEWS = ["dc", "gbm", "mkt", "ens"]


@dataclass
class Artifacts:
    league: str
    trained_at: str
    last_result: str
    elo_model: elo.EloOutcomeModel
    dc_fit: dixon_coles.DCFit | None
    gbm: GBMModel | None
    gbm_mkt: GBMModel | None
    weights: dict
    pure_features: list[str]
    market_features: list[str]
    track_record: dict = field(default_factory=dict)


def artifacts_path(code: str):
    return config.MODELS_DIR / f"{code.upper()}.pkl"


def outputs(code: str) -> dict:
    d = config.OUTPUT_DIR
    c = code.upper()
    return {
        "backtest": d / f"{c}_backtest.parquet",
        "predictions": d / f"{c}_predictions.parquet",
        "track_record": d / f"{c}_track_record.json",
        "features": d / f"{c}_features.parquet",
    }


def _final_gbm(X: pd.DataFrame, y: np.ndarray, seasons: pd.Series, features: list[str]) -> GBMModel | None:
    """Find rounds + temperature with the last season as validation, then refit on everything."""
    probe = bt_mod._gbm_fit(X, y, seasons, features)
    if probe is None:
        return None
    final = GBMModel(features)
    final.fit(X, y, rounds=int(probe.best_iteration * 1.1))
    final.temperature = probe.temperature
    return final


def train(league: League, matches: pd.DataFrame | None = None, verbose: bool = True) -> tuple[Artifacts, bt_mod.Prepared]:
    config.ensure_dirs()
    matches = store.load_matches(league.code) if matches is None else matches
    if matches.empty:
        raise RuntimeError(f"No data for {league.code}. Run: ideam fetch {league.code}")
    prep = bt_mod.prepare(league, matches, verbose)
    bt, info = bt_mod.run(prep, verbose)

    # ---- track record (proper scoring rules + betting backtests, per view) ----
    views = [v for v in ["base"] + bt_mod.PANEL_VIEWS if f"p_{v}_h" in bt]
    track: dict = {"views": metrics.VIEWS, "info": info}
    if not bt.empty:
        track["overall"] = metrics.metrics_table(bt, views).reset_index().to_dict("records")
        track["common"] = metrics.metrics_table(bt, ["elo", "dc", "gbm", "ens"], common=True).reset_index().to_dict("records")
        with_odds = bt.dropna(subset=["p_book_open_h"])
        if len(with_odds) > 100:
            track["vs_market"] = metrics.metrics_table(
                with_odds, ["elo", "dc", "gbm", "mkt", "ens", "book_open", "book_close"], common=False
            ).reset_index().to_dict("records")
            track["betting"] = betting.simulate_all(bt, BET_VIEWS).reset_index().to_dict("records")
        recent = bt[bt["season"].isin(prep.test_seasons[-2:])]
        track["recent"] = metrics.metrics_table(recent, views).reset_index().to_dict("records")
        track["by_season"] = [
            {"season": s, **{v: metrics.score(metrics.probs_of(g, v), g["y"].to_numpy(int))["rps"]
                            for v in views}}
            for s, g in bt.groupby("season")
        ]
        bt.to_parquet(outputs(league.code)["backtest"])

    # ---- final production models on all played matches ----
    m, X = prep.matches, prep.X
    played = m["hg"].notna().to_numpy()
    y = np.full(len(m), -1)
    y[played] = outcome(m.loc[played, "hg"], m.loc[played, "ag"])
    rank = m["season"].map(bt_mod._season_start_year)
    trainable = played & (rank >= bt_mod._season_start_year(prep.train_from)).to_numpy() & X["p_dc_h"].notna().to_numpy()
    em = elo.EloOutcomeModel().fit(X.loc[played, "elo_diff"], m.loc[played, "hg"], m.loc[played, "ag"])
    gbm = _final_gbm(X[trainable], y[trainable], m.loc[trainable, "season"], prep.pure_features)
    has_odds = X[MARKET_FEATURES[:3]].notna().all(axis=1).to_numpy()
    gbm_mkt = None
    if (trainable & has_odds).sum() >= 500:
        sel = trainable & has_odds
        gbm_mkt = _final_gbm(X[sel], y[sel], m.loc[sel, "season"], prep.market_features)
    if not bt.empty:
        recent = bt[bt["season"].isin(prep.test_seasons[-4:])]
        w = ensemble.fit_weights([metrics.probs_of(recent, v) for v in bt_mod.ENSEMBLE_PARTS], recent["y"].to_numpy(int))
    else:
        w = np.array([0.2, 0.4, 0.4])
    art = Artifacts(
        league=league.code,
        trained_at=datetime.now().isoformat(timespec="seconds"),
        last_result=str(m.loc[played, "date"].max().date()),
        elo_model=em, dc_fit=prep.dc_final, gbm=gbm, gbm_mkt=gbm_mkt,
        weights=dict(zip(bt_mod.ENSEMBLE_PARTS, map(float, w))),
        pure_features=prep.pure_features, market_features=prep.market_features,
        track_record=track,
    )
    with open(artifacts_path(league.code), "wb") as fh:
        pickle.dump(art, fh)
    with open(outputs(league.code)["track_record"], "w") as fh:
        json.dump(track, fh, indent=1, default=str)
    if verbose:
        if "common" in track:
            print(pd.DataFrame(track["common"]).set_index("view")[["n", "rps", "logloss", "accuracy"]].round(4))
        print(f"Saved models to {artifacts_path(league.code)}")
    return art, prep


def load_artifacts(code: str) -> Artifacts:
    # Pickle is safe here: these files are only ever written by train() on this machine.
    # Never load model files you downloaded from someone else.
    with open(artifacts_path(code), "rb") as fh:
        return pickle.load(fh)


def panel(art: Artifacts, prep_or_X: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """All model views side by side for the given rows (feature frame aligned with matches)."""
    X = prep_or_X
    out = pd.DataFrame(index=X.index)
    out[["p_elo_h", "p_elo_d", "p_elo_a"]] = art.elo_model.predict_proba(X["elo_diff"])
    out[["p_dc_h", "p_dc_d", "p_dc_a"]] = X[["p_dc_h", "p_dc_d", "p_dc_a"]].to_numpy()
    out["p_dc_o25"] = X["p_dc_o25"]
    out["dc_lambda_h"] = X["dc_lambda_h"]
    out["dc_lambda_a"] = X["dc_lambda_a"]
    if art.gbm is not None:
        out[["p_gbm_h", "p_gbm_d", "p_gbm_a"]] = art.gbm.predict_proba(X)
    else:
        out[["p_gbm_h", "p_gbm_d", "p_gbm_a"]] = np.nan
    out[["p_mkt_h", "p_mkt_d", "p_mkt_a"]] = np.nan
    has_odds = X[MARKET_FEATURES[:3]].notna().all(axis=1)
    if art.gbm_mkt is not None and has_odds.any():
        out.loc[has_odds, ["p_mkt_h", "p_mkt_d", "p_mkt_a"]] = art.gbm_mkt.predict_proba(X[has_odds])
    w = np.array([art.weights[v] for v in bt_mod.ENSEMBLE_PARTS])
    out[["p_ens_h", "p_ens_d", "p_ens_a"]] = ensemble.blend(
        [out[[f"p_{v}_h", f"p_{v}_d", f"p_{v}_a"]].to_numpy() for v in bt_mod.ENSEMBLE_PARTS], w)
    out[["p_book_open_h", "p_book_open_d", "p_book_open_a", "p_book_open_o25"]] = X[
        ["p_book_open_h", "p_book_open_d", "p_book_open_a", "p_book_open_o25"]].to_numpy()
    out = add_agreement(out)
    ids = matches.loc[X.index, ["match_id", "league", "season", "date", "home", "away", "neutral", "tournament",
                                "source_event_id", "odds_h_open", "odds_d_open", "odds_a_open",
                                "odds_o25_open", "odds_u25_open", "odds_h_close", "odds_d_close", "odds_a_close"]]
    return ids.join(out)


def add_agreement(out: pd.DataFrame) -> pd.DataFrame:
    """How much do the independent views agree? Mean total-variation distance to the ensemble."""
    views = [v for v in ["elo", "dc", "gbm", "mkt", "book_open"] if f"p_{v}_h" in out]
    ens = out[["p_ens_h", "p_ens_d", "p_ens_a"]].to_numpy()
    dists = []
    for v in views:
        p = out[[f"p_{v}_h", f"p_{v}_d", f"p_{v}_a"]].to_numpy()
        dists.append(0.5 * np.abs(p - ens).sum(axis=1))
    d = np.nanmean(np.vstack(dists), axis=0)
    out["disagreement"] = d
    out["agreement"] = np.select([d < 0.04, d < 0.08], ["high", "moderate"], "low")
    picks = np.array(["home", "draw", "away"])[np.nan_to_num(ens, nan=0).argmax(axis=1)]
    out["pick"] = picks
    return out


def predict(league: League, art: Artifacts | None = None, prep: bt_mod.Prepared | None = None,
            verbose: bool = True) -> pd.DataFrame:
    """Predict every scheduled fixture and save predictions + their feature rows."""
    art = art or load_artifacts(league.code)
    if prep is None:
        prep = bt_mod.prepare(league, store.load_matches(league.code), verbose)
    m = prep.matches
    sched = m[m["hg"].isna()].index
    if len(sched) == 0:
        if verbose:
            print("No scheduled fixtures in the data. Use the dashboard's custom matchup instead.")
        preds = pd.DataFrame()
    else:
        preds = panel(art, prep.X.loc[sched], m)
    o = outputs(league.code)
    preds.to_parquet(o["predictions"])
    prep.X.loc[sched].to_parquet(o["features"])
    if verbose and not preds.empty:
        show = preds[["date", "home", "away", "p_ens_h", "p_ens_d", "p_ens_a", "pick", "agreement"]].copy()
        show[["p_ens_h", "p_ens_d", "p_ens_a"]] = (show[["p_ens_h", "p_ens_d", "p_ens_a"]] * 100).round(1)
        print(show.to_string(index=False))
    return preds


def predict_custom(league: League, art: Artifacts, matches: pd.DataFrame, home: str, away: str,
                   date=None, neutral: bool = False, odds: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Any matchup on any date (e.g. a World Cup game) using the saved models."""
    date = pd.Timestamp(date or pd.Timestamp.today().normalize())
    row = {"league": league.code, "date": date, "home": home, "away": away, "neutral": neutral,
           "season": league.season_of(date), "status": "SCHEDULED", "tournament": "Custom"}
    for k, v in (odds or {}).items():
        row[k] = v
    m = conform(pd.concat([matches[matches["date"] < date], pd.DataFrame([row])], ignore_index=True))
    from .data.sources.international import tournament_importance
    imp = tournament_importance(m["tournament"]) if league.source == "international" else None
    elo_df, _ = elo.compute(m, league.elo, imp)
    target = m.index[(m["home"] == home) & (m["away"] == away) & (m["date"] == date) & m["hg"].isna()]
    dc_df = pd.DataFrame(index=m.index)
    if art.dc_fit is not None:
        dc_df = dixon_coles.predict(art.dc_fit, m.loc[target]).reindex(m.index)
    # features only need recent history for speed
    recent = m.index[m["date"] >= date - pd.Timedelta(days=3 * 365)]
    X = feats.build(m.loc[recent], elo_df.loc[recent], dc_df.loc[recent], imp.loc[recent] if imp is not None else None)
    return panel(art, X.loc[target], m), X.loc[target]


def run(code: str, fetch: bool = True, verbose: bool = True) -> pd.DataFrame:
    league = get_league(code)
    if fetch:
        from .data.fetch import update_league
        update_league(league)
    art, prep = train(league, verbose=verbose)
    return predict(league, art, prep, verbose)

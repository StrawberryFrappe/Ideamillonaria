"""Walk-forward backtest.

For each test season S every model is trained only on seasons before S and then
predicts S. Dixon-Coles is refitted every ``refit_days`` inside the season, Elo is
updated match by match (it is sequential by nature). Nothing from season S - or from
later in the same matchweek - leaks into a prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import League
from ..data.schema import outcome
from ..data.sources.international import tournament_importance
from ..models import dixon_coles, elo, ensemble
from ..models import features as feats
from ..models.gbm import GBMModel
from ..models.market import MARKET_FEATURES

PANEL_VIEWS = ["elo", "dc", "gbm", "mkt", "ens", "book_open", "book_close"]
ENSEMBLE_PARTS = ["elo", "dc", "gbm"]


@dataclass
class Prepared:
    league: League
    matches: pd.DataFrame
    elo: pd.DataFrame
    elo_state: elo.EloState
    dc: pd.DataFrame
    dc_final: dixon_coles.DCFit | None
    X: pd.DataFrame
    importance: pd.Series | None
    seasons: list[str]
    train_from: str
    test_seasons: list[str]
    pure_features: list[str] = field(default_factory=list)
    market_features: list[str] = field(default_factory=list)


def _season_start_year(label: str) -> int:
    return int(str(label)[:4])


def prepare(league: League, matches: pd.DataFrame, verbose: bool = True) -> Prepared:
    m = matches.sort_values(["date", "home"], kind="stable").reset_index(drop=True)
    importance = tournament_importance(m["tournament"]) if league.source == "international" else None
    elo_df, state = elo.compute(m, league.elo, importance)

    played = m[m["hg"].notna()]
    seasons = sorted(played["season"].unique(), key=_season_start_year)
    if len(seasons) < 2:
        raise ValueError(f"{league.code}: need at least 2 seasons of results, have {len(seasons)}")
    if league.backtest_from is not None:
        test_seasons = [s for s in seasons if _season_start_year(s) >= league.backtest_from]
        train_from_year = league.backtest_from - 18
        train_from = next((s for s in seasons if _season_start_year(s) >= train_from_year), seasons[1])
    else:
        test_seasons = seasons[1 + league.min_train_seasons:]
        train_from = seasons[1]
    dc_start = m.loc[m["season"] == train_from, "date"].min()
    if verbose:
        print(f"{league.code}: {len(played)} results, {len(seasons)} seasons, "
              f"testing {test_seasons[0] if test_seasons else '-'}..{test_seasons[-1] if test_seasons else '-'}")

    dc_df, _ = dixon_coles.rolling_predict(
        m[m["date"] >= dc_start - pd.Timedelta(days=league.dc.window_days)],
        league.dc.xi, league.dc.ridge, league.dc.refit_days, league.dc.window_days, dc_start, verbose,
    )
    dc_df = dc_df[~dc_df.index.duplicated(keep="last")] if not dc_df.empty else dc_df
    # production fit on everything played, used for scheduled fixtures
    dc_final = dixon_coles.fit(played[played["date"] > played["date"].max() - pd.Timedelta(days=league.dc.window_days)],
                               played["date"].max() + pd.Timedelta(days=1), league.dc.xi, league.dc.ridge)
    sched = m[m["hg"].isna()]
    if dc_final is not None and not sched.empty:
        dc_sched = dixon_coles.predict(dc_final, sched)
        dc_df = pd.concat([dc_df.drop(index=sched.index, errors="ignore"), dc_sched])

    X = feats.build(m, elo_df, dc_df.reindex(m.index) if not dc_df.empty else None, importance)
    return Prepared(league, m, elo_df, state, dc_df.reindex(m.index), dc_final, X, importance,
                    seasons, train_from, test_seasons, feats.pure_feature_names(X),
                    feats.market_feature_names(X))


def _book(m: pd.DataFrame, X: pd.DataFrame, when: str) -> pd.DataFrame:
    if when == "open":
        b = X[["p_book_open_h", "p_book_open_d", "p_book_open_a", "p_book_open_o25"]]
    else:
        from ..models.market import book_probs
        b = book_probs(m, "close")[["p_book_close_h", "p_book_close_d", "p_book_close_a", "p_book_close_o25"]]
    return b


def _gbm_fit(X: pd.DataFrame, y: np.ndarray, seasons: pd.Series, features: list[str]) -> GBMModel | None:
    labels = sorted(seasons.unique(), key=_season_start_year)
    if len(labels) < 2 or len(X) < 300:
        return None
    val = seasons == labels[-1]
    model = GBMModel(features)
    model.fit(X[~val], y[~val], X[val], y[val])
    return model


def run(prep: Prepared, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    m, X = prep.matches, prep.X
    played_mask = m["hg"].notna().to_numpy()
    y_all = np.full(len(m), -1)
    y_all[played_mask] = outcome(m.loc[played_mask, "hg"], m.loc[played_mask, "ag"])
    season_rank = m["season"].map(_season_start_year)
    train_from_rank = _season_start_year(prep.train_from)
    has_dc = X["p_dc_h"].notna().to_numpy()
    has_odds = X[MARKET_FEATURES[:3]].notna().all(axis=1).to_numpy()

    results = []
    info: dict = {"weights": {}, "gbm_iterations": {}, "temperature": {}}
    history = []  # previous test seasons' OOS predictions for ensemble weights
    for S in prep.test_seasons:
        rank = _season_start_year(S)
        test = (m["season"] == S).to_numpy() & played_mask
        if test.sum() == 0:
            continue
        past = played_mask & (season_rank < rank).to_numpy()
        train = past & (season_rank >= train_from_rank).to_numpy() & has_dc

        out = pd.DataFrame(index=m.index[test])
        # Elo -> probabilities
        em = elo.EloOutcomeModel().fit(X.loc[past, "elo_diff"], m.loc[past, "hg"], m.loc[past, "ag"])
        out[["p_elo_h", "p_elo_d", "p_elo_a"]] = em.predict_proba(X.loc[test, "elo_diff"])
        # Dixon-Coles (already walk-forward)
        out[["p_dc_h", "p_dc_d", "p_dc_a", "p_dc_o25", "p_dc_btts"]] = prep.dc.loc[test, [
            "p_dc_h", "p_dc_d", "p_dc_a", "p_dc_o25", "p_dc_btts"]].to_numpy()
        # Stats model (pure)
        g = _gbm_fit(X[train], y_all[train], m.loc[train, "season"], prep.pure_features)
        if g is not None:
            out[["p_gbm_h", "p_gbm_d", "p_gbm_a"]] = g.predict_proba(X[test])
            info["gbm_iterations"][S] = g.best_iteration
            info["temperature"][S] = g.temperature
        else:
            out[["p_gbm_h", "p_gbm_d", "p_gbm_a"]] = np.nan
        # Market-aware model: same features + opening odds, only where odds exist
        tr_mkt = train & has_odds
        te_mkt = test & has_odds
        out[["p_mkt_h", "p_mkt_d", "p_mkt_a"]] = np.nan
        if tr_mkt.sum() >= 500 and te_mkt.sum() > 0:
            gm = _gbm_fit(X[tr_mkt], y_all[tr_mkt], m.loc[tr_mkt, "season"], prep.market_features)
            if gm is not None:
                out.loc[m.index[te_mkt], ["p_mkt_h", "p_mkt_d", "p_mkt_a"]] = gm.predict_proba(X[te_mkt])
        # Base rates
        base = np.bincount(y_all[past], minlength=3) / max(past.sum(), 1)
        out[["p_base_h", "p_base_d", "p_base_a"]] = base
        # Ensemble weights: fit on earlier test seasons' out-of-sample predictions
        if history:
            hist = pd.concat(history[-4:])
            w = ensemble.fit_weights([hist[[f"p_{v}_h", f"p_{v}_d", f"p_{v}_a"]].to_numpy() for v in ENSEMBLE_PARTS],
                                     hist["y"].to_numpy(int))
        else:
            w = np.array([0.2, 0.4, 0.4])
        info["weights"][S] = dict(zip(ENSEMBLE_PARTS, map(float, w)))
        out[["p_ens_h", "p_ens_d", "p_ens_a"]] = ensemble.blend(
            [out[[f"p_{v}_h", f"p_{v}_d", f"p_{v}_a"]].to_numpy() for v in ENSEMBLE_PARTS], w)
        out["y"] = y_all[test]
        history.append(out)
        results.append(out)
        if verbose:
            from .metrics import score
            s_ens = score(out[["p_ens_h", "p_ens_d", "p_ens_a"]].to_numpy(), out["y"].to_numpy())
            print(f"  season {S}: {test.sum()} matches, ensemble RPS {s_ens['rps']:.4f} "
                  f"acc {s_ens['accuracy']:.3f}  weights {np.round(w, 2).tolist()}")

    if not results:
        return pd.DataFrame(), info
    bt = pd.concat(results)
    ids = m.loc[bt.index, ["match_id", "league", "season", "date", "home", "away", "hg", "ag", "neutral",
                            "odds_h_open", "odds_d_open", "odds_a_open", "odds_h_close", "odds_d_close",
                            "odds_a_close", "odds_o25_open", "odds_u25_open", "odds_o25_close", "odds_u25_close"]]
    bt = ids.join(bt)
    bt = bt.join(_book(m, X, "open").loc[bt.index]).join(_book(m, X, "close").loc[bt.index])
    return bt.sort_values("date", kind="stable"), info

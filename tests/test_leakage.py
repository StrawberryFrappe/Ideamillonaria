"""No-lookahead and model-purity guarantees."""

import numpy as np
import pandas as pd

from ideamillonaria.config import get_league
from ideamillonaria.data.schema import MARKET_MARKERS
from ideamillonaria.evaluate import backtest
from ideamillonaria.models import dixon_coles, elo
from ideamillonaria.models import features as feats


def _features(m, league):
    e, _ = elo.compute(m, league.elo)
    return feats.build(m, e, None)


def test_features_do_not_see_the_future(synthetic):
    df, _ = synthetic
    league = get_league("SYN")
    df = df[df["hg"].notna()].reset_index(drop=True)
    cutoff = df["date"].iloc[len(df) // 2]
    full = _features(df, league)
    # drop everything on/after the cutoff, but keep the cutoff matches' rows as unplayed fixtures
    truncated = df[df["date"] <= cutoff].copy()
    on_cut = truncated["date"] == cutoff
    stat_cols = [c for c in truncated.columns if c.startswith(("hg", "ag", "h_", "a_")) or "close" in c]
    truncated.loc[on_cut, stat_cols] = np.nan
    trunc_feats = _features(truncated, league)
    cols = [c for c in full.columns if not any(mk in c for mk in MARKET_MARKERS)]
    idx = truncated.index
    pd.testing.assert_frame_equal(full.loc[idx, cols], trunc_feats.loc[idx, cols], check_dtype=False)


def test_dixon_coles_walk_forward_is_out_of_sample(synthetic):
    df, _ = synthetic
    df = df[df["hg"].notna()].reset_index(drop=True)
    start = df["date"].iloc[len(df) // 3]
    cutoff = df["date"].iloc[2 * len(df) // 3]
    full, _ = dixon_coles.rolling_predict(df, 0.002, 1.0, 7, 1000, start)
    part, _ = dixon_coles.rolling_predict(df[df["date"] < cutoff], 0.002, 1.0, 7, 1000, start)
    common = part.index
    assert np.allclose(full.loc[common, "p_dc_h"], part.loc[common, "p_dc_h"])


def test_pure_models_never_see_odds(synthetic):
    df, _ = synthetic
    prep = backtest.prepare(get_league("SYN"), df, verbose=False)
    assert not [f for f in prep.pure_features if any(mk in f for mk in MARKET_MARKERS)]
    assert "p_book_open_h" in prep.market_features
    # and never closing odds - not even the market-aware model
    assert not [f for f in prep.market_features if "close" in f]

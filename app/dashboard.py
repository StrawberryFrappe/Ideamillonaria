"""Ideamillonaria dashboard: ``streamlit run app/dashboard.py`` (or ``ideam dashboard``)."""

from __future__ import annotations

import json
import warnings
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ideamillonaria import journal, pipeline
from ideamillonaria.config import LEAGUES, get_league
from ideamillonaria.data import store
from ideamillonaria.evaluate import betting, metrics
from ideamillonaria.report import context, explain, markets

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Ideamillonaria", page_icon="⚽", layout="wide")

# Outcome colours (validated categorical slots 1-3: blue, orange, aqua). Always paired
# with direct % labels, so colour is never the only cue.
C_HOME, C_DRAW, C_AWAY = "#2a78d6", "#eb6834", "#1baf7a"
VIEW_COLORS = {"elo": "#2a78d6", "dc": "#eb6834", "gbm": "#1baf7a", "mkt": "#eda100",
               "ens": "#e87ba4", "book_open": "#008300", "book_close": "#6250d6", "base": "#9a9890"}
VIEW_LABEL = {
    "elo": "Elo rating", "dc": "Goals model (Dixon-Coles)", "gbm": "Stats model (no odds)",
    "mkt": "Stats + market model", "ens": "Ensemble", "book_open": "Bookmakers (fair)",
    "book_close": "Bookmakers closing", "base": "Base rates",
}
OUT_NAMES = ["home", "draw", "away"]


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_matches(code: str, _mtime: float) -> pd.DataFrame:
    return store.load_matches(code)


@st.cache_resource(show_spinner=False)
def load_art(code: str, _mtime: float):
    return pipeline.load_artifacts(code)


@st.cache_data(show_spinner=False)
def load_parquet(path: str, _mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def mtime(p) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def trained_leagues() -> list[str]:
    return [c for c in LEAGUES if pipeline.artifacts_path(c).exists()]


# ----------------------------------------------------------------------------- charts
def panel_chart(row: pd.Series) -> go.Figure:
    views = [v for v in ["ens", "elo", "dc", "gbm", "mkt", "book_open"] if pd.notna(row.get(f"p_{v}_h"))]
    labels = [VIEW_LABEL[v] for v in views]
    fig = go.Figure()
    for k, (name, col) in enumerate(zip([row["home"], "Draw", row["away"]], [C_HOME, C_DRAW, C_AWAY])):
        vals = [row[f"p_{v}_{'hda'[k]}"] for v in views]
        fig.add_bar(y=labels, x=vals, name=name, orientation="h", marker_color=col,
                    marker_line_color="rgba(255,255,255,0.9)", marker_line_width=2,
                    text=[f"{x:.0%}" for x in vals], textposition="inside", insidetextanchor="middle",
                    hovertemplate=f"{name}: %{{x:.1%}}<br>%{{y}}<extra></extra>")
    fig.update_layout(barmode="stack", height=60 + 42 * len(views), margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(tickformat=".0%", range=[0, 1], showgrid=False),
                      yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.12, x=0, traceorder="normal"),
                      uniformtext_minsize=10, uniformtext_mode="hide")
    return fig


def heatmap(lam: float, mu: float, rho: float, home: str, away: str) -> go.Figure:
    mat = markets.matrix(lam, mu, rho, 5)
    text = [[f"{v:.0%}" if v >= 0.005 else "" for v in r] for r in mat]
    fig = go.Figure(go.Heatmap(
        z=mat, x=[str(i) for i in range(6)], y=[str(i) for i in range(6)], text=text, texttemplate="%{text}",
        colorscale=[[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]], showscale=False, xgap=2, ygap=2,
        hovertemplate=f"{home} %{{y}} - %{{x}} {away}<br>probability %{{z:.1%}}<extra></extra>"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title=f"{away} goals", side="top"), yaxis=dict(title=f"{home} goals",
                                                                                autorange="reversed"))
    return fig


# ----------------------------------------------------------------------------- report
def render_report(code: str, row: pd.Series, x_row: pd.Series | None, matches: pd.DataFrame, art, key: str):
    league = get_league(code)
    home, away = row["home"], row["away"]
    when = pd.Timestamp(row["date"])
    st.subheader(f"{home} vs {away}")
    st.caption(f"{league.name} · {when.date()}" + (" · neutral venue" if row.get("neutral") else ""))

    rho = art.dc_fit.rho if art.dc_fit is not None else 0.0
    h = explain.headline(row, rho)
    cols = st.columns(5)
    cols[0].metric("Pick", h["pick"], h["pick_prob"], delta_color="off", delta_arrow="off",
                   help="Ensemble's most likely result. Full split: " + h["split"])
    cols[1].metric("Expected goals", h["xg"], help="Goals model's expected goals for each side.")
    cols[2].metric("Likely score", h["score"], h["score_prob"], delta_color="off", delta_arrow="off",
                   help="Single most likely scoreline; see the heatmap below for the rest.")
    cols[3].metric("Agreement", h["agreement"], h["spread"], delta_color="off", delta_arrow="off",
                   help="How closely Elo, goals, stats, stats+market and bookmakers agree. "
                        "Low agreement = genuinely uncertain match.")
    cols[4].metric("Value", h["value"], h["value_edge"], delta_color="off", delta_arrow="off",
                   help="Best ensemble edge against the stored odds (model probability x odds - 1). "
                        "Small edges (<5%) are usually noise.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Views")
        st.plotly_chart(panel_chart(row), use_container_width=True, key=f"panel-{key}")
        tr = pd.DataFrame(art.track_record.get("recent") or art.track_record.get("overall") or [])
        if not tr.empty:
            tr = tr[tr["view"].isin(["elo", "dc", "gbm", "mkt", "ens", "book_open"]) & (tr["n"] > 0)]
            tr = tr.assign(view=tr["view"].map(VIEW_LABEL))[["view", "n", "rps", "accuracy"]]
            st.dataframe(tr.style.format({"rps": "{:.4f}", "accuracy": "{:.1%}"}), hide_index=True,
                         use_container_width=True,
                         column_config={"rps": st.column_config.NumberColumn(
                             "RPS", help="Last two backtested seasons, out of sample. Lower is better.")})
    with right:
        st.markdown("#### Odds & value")
        stored = [row.get(f"odds_{o}_open") for o in "hda"]
        fair = [min(100.0, max(1.01, round(1 / row[f"p_ens_{o}"], 2))) for o in "hda"]
        defaults = [float(s) if pd.notna(s) else f for s, f in zip(stored, fair)]
        c1, c2, c3 = st.columns(3)
        o_h = c1.number_input(f"{home}", 1.01, 100.0, defaults[0], 0.01, key=f"oh-{key}",
                              help="Stored opening odds if available, otherwise the ensemble's fair odds. "
                                   "Type your bookmaker's prices.")
        o_d = c2.number_input("Draw", 1.01, 100.0, defaults[1], 0.01, key=f"od-{key}")
        o_a = c3.number_input(f"{away}", 1.01, 100.0, defaults[2], 0.01, key=f"oa-{key}")
        odds = np.array([o_h, o_d, o_a])
        bankroll = st.session_state.get("bankroll", 100.0)
        rows = []
        for v in ["ens", "gbm", "mkt", "dc", "elo"]:
            p = np.array([row.get(f"p_{v}_{o}", np.nan) for o in "hda"], float)
            if np.isnan(p).any():
                continue
            for k in range(3):
                e = p[k] * odds[k] - 1
                rows.append({"view": VIEW_LABEL[v], "bet": [home, "Draw", away][k], "model prob": p[k],
                             "fair odds": 1 / p[k], "edge": e,
                             "¼-Kelly stake": betting.suggested_stake(p[k], odds[k], bankroll) if e > 0 else 0.0})
        vt = pd.DataFrame(rows)
        margin = (1 / odds).sum() - 1
        st.caption(f"Margin {margin:.1%}" + (" · fair odds shown" if all(pd.isna(s) for s in stored) else ""))
        if not vt.empty:
            show = vt[vt["edge"] > 0].sort_values("edge", ascending=False)
            if show.empty:
                st.caption("No value at these prices.")
            else:
                st.dataframe(show.style.format({"model prob": "{:.1%}", "fair odds": "{:.2f}",
                                                "edge": "{:+.1%}", "¼-Kelly stake": "{:.2f}"}),
                             hide_index=True, use_container_width=True)
        with st.expander("Log a bet in my journal"):
            with st.form(f"bet-{key}"):
                sel = st.selectbox("Selection", [f"{home} win", "Draw", f"{away} win", "Over 2.5", "Under 2.5"])
                b_odds = st.number_input("Odds taken", 1.01, 100.0, 2.0, 0.01)
                stake = st.number_input("Stake", 0.0, 1e6, 1.0, 0.5)
                view = st.selectbox("Which view were you following?", ["ens", "gbm", "mkt", "dc", "elo", "own"])
                if st.form_submit_button("Save bet"):
                    idx = [f"{home} win", "Draw", f"{away} win", "Over 2.5", "Under 2.5"].index(sel)
                    market, selection = (("1X2", "hda"[idx]) if idx < 3 else ("OU25", ["over", "under"][idx - 3]))
                    prob = row.get(f"p_{view}_{'hda'[idx]}") if idx < 3 else (
                        row.get("p_dc_o25") if idx == 3 else 1 - row.get("p_dc_o25", np.nan))
                    journal.add_bet(code, when, home, away, market, selection, b_odds, stake, view,
                                    None if prob is None or pd.isna(prob) else float(prob))
                    st.success("Saved. It settles automatically after the match (ideam update).")

    if art.dc_fit is not None and pd.notna(row.get("dc_lambda_h")):
        st.markdown("#### Scorelines & markets")
        a, b = st.columns([2, 3])
        with a:
            st.plotly_chart(heatmap(row["dc_lambda_h"], row["dc_lambda_a"], art.dc_fit.rho, home, away),
                            use_container_width=True, key=f"heat-{key}")
            st.dataframe(markets.top_scorelines(row["dc_lambda_h"], row["dc_lambda_a"], art.dc_fit.rho, 8)
                         .style.format({"probability": "{:.1%}", "fair_odds": "{:.1f}"}), hide_index=True,
                         use_container_width=True)
        with b:
            mk = markets.markets(row["dc_lambda_h"], row["dc_lambda_a"], art.dc_fit.rho, home, away)
            st.dataframe(mk.style.format({"probability": "{:.1%}", "fair_odds": "{:.2f}"}), hide_index=True,
                         use_container_width=True, height=520)

    if x_row is not None and art.gbm is not None:
        st.markdown("#### Key factors", help="Largest SHAP contributions in the stats model for its pick.")
        contrib = art.gbm.contributions(x_row.to_frame().T)[0]
        k = int(np.nanargmax([row["p_gbm_h"], row["p_gbm_d"], row["p_gbm_a"]]))
        rs = explain.reasons(contrib, art.gbm.features, x_row, k, home, away, top=6)
        st.caption(f"Stats model → {[home + ' win', 'draw', away + ' win'][k]}")
        if not rs.empty:
            st.dataframe(rs.assign(effect=rs["effect"].map({"for": "▲ for", "against": "▼ against"}))
                         [["effect", "reason"]], hide_index=True, use_container_width=True)

    st.markdown("#### Context")
    c1, c2 = st.columns(2)
    for col, team in ((c1, home), (c2, away)):
        with col:
            rd = context.rest_days(matches, team, when)
            st.markdown(f"**{team}** - last 10 matches" + (f" · {rd:.0f} days rest" if pd.notna(rd) else ""))
            st.dataframe(context.team_matches(matches, team, when, 10).dropna(axis=1, how="all"), hide_index=True,
                         use_container_width=True)
    h2h = context.head_to_head(matches, home, away, when)
    st.markdown("**Head to head**")
    if h2h.empty:
        st.caption("No previous meetings.")
    else:
        st.dataframe(h2h, hide_index=True, use_container_width=True)
    if league.source != "international":
        table = context.standings(matches, row["season"], when)
        if not table.empty:
            with st.expander(f"League table {row['season']}"):
                st.dataframe(table.style.apply(
                    lambda r: ["font-weight: bold" if r.name in (home, away) else "" for _ in r], axis=1),
                    use_container_width=True)
    if league.source == "espn" and row.get("source_event_id"):
        if st.button("Fetch lineups & injuries (ESPN)", key=f"espn-{key}"):
            from ideamillonaria.data.sources import espn
            extras = espn.fetch_summary(league, row["source_event_id"])
            st.json(extras or {"info": "nothing published yet"})


# ----------------------------------------------------------------------------- pages
def page_upcoming(code: str, art, matches: pd.DataFrame):
    o = pipeline.outputs(code)
    if not o["predictions"].exists():
        st.info("No predictions yet. Run `ideam update " + code + "`.")
        return
    preds = load_parquet(str(o["predictions"]), mtime(o["predictions"]))
    if preds.empty:
        st.info("No upcoming fixtures in the data. Use **Custom matchup** to predict any game.")
        return
    feats_df = load_parquet(str(o["features"]), mtime(o["features"]))
    preds = preds.sort_values("date")
    tbl = pd.DataFrame({
        "date": preds["date"].dt.date,
        "match": preds["home"] + " vs " + preds["away"],
        "home %": preds["p_ens_h"], "draw %": preds["p_ens_d"], "away %": preds["p_ens_a"],
        "pick": preds["pick"], "agreement": preds["agreement"],
        "ensemble value": [
            (lambda v: f"{['H', 'D', 'A'][v[0]['outcome']]} @ {v[0]['odds']:.2f} ({v[0]['edge']:+.0%})"
             if v else "")(explain.best_value(r, views=("ens",), min_edge=0.05)) for _, r in preds.iterrows()],
    })
    st.caption(f"Trained {art.trained_at[:10]} · results to {art.last_result} · ensemble %")
    st.dataframe(tbl.style.format({"home %": "{:.0%}", "draw %": "{:.0%}", "away %": "{:.0%}"}),
                 hide_index=True, use_container_width=True)
    choice = st.selectbox("Open match report", range(len(preds)),
                          format_func=lambda i: f"{tbl['date'].iloc[i]}  {tbl['match'].iloc[i]}")
    row = preds.iloc[choice]
    x_row = feats_df.loc[row.name] if row.name in feats_df.index else None
    st.divider()
    render_report(code, row, x_row, matches, art, key=f"up-{choice}")


def page_custom(code: str, art, matches: pd.DataFrame):
    league = get_league(code)
    recent = matches[matches["date"] >= matches["date"].max() - pd.Timedelta(days=3 * 365)]
    teams = sorted(set(recent["home"]) | set(recent["away"]))
    c1, c2, c3, c4 = st.columns([3, 3, 2, 1])
    home = c1.selectbox("Home team", teams, index=0)
    away = c2.selectbox("Away team", teams, index=min(1, len(teams) - 1))
    when = c3.date_input("Date", date.today())
    neutral = c4.checkbox("Neutral", value=league.source == "international")
    if home == away:
        st.warning("Pick two different teams.")
        return
    if st.button("Predict", type="primary"):
        with st.spinner("Computing features..."):
            p, X = pipeline.predict_custom(league, art, matches, home, away, when, neutral)
        st.session_state["custom"] = (p.iloc[0], X.iloc[0])
    if "custom" in st.session_state:
        row, x_row = st.session_state["custom"]
        if row["home"] == home and row["away"] == away:
            render_report(code, row, x_row, matches, art, key="custom")


def page_track(code: str, art):
    tr = art.track_record
    st.caption("Out-of-sample (walk-forward) · lower RPS / log-loss is better")
    if "common" in tr:
        st.markdown("##### Accuracy (same matches for every view)")
        df = pd.DataFrame(tr["common"])
        df = df[df["n"] > 0]
        df["view"] = df["view"].map(VIEW_LABEL)
        st.dataframe(df[["view", "n", "rps", "logloss", "brier", "accuracy"]].style.format(
            {"rps": "{:.4f}", "logloss": "{:.4f}", "brier": "{:.4f}", "accuracy": "{:.1%}"}),
            hide_index=True, use_container_width=True)
    if "vs_market" in tr:
        st.markdown("##### Against the bookmakers (matches with odds)")
        df = pd.DataFrame(tr["vs_market"])
        df = df[df["n"] > 0]
        df["view"] = df["view"].map(VIEW_LABEL)
        st.dataframe(df[["view", "n", "rps", "logloss", "accuracy"]].style.format(
            {"rps": "{:.4f}", "logloss": "{:.4f}", "accuracy": "{:.1%}"}), hide_index=True, use_container_width=True)
    if tr.get("by_season"):
        bs = pd.DataFrame(tr["by_season"]).set_index("season")
        fig = go.Figure()
        for v in [c for c in ["ens", "gbm", "dc", "elo", "book_open", "base"] if c in bs]:
            fig.add_scatter(x=bs.index, y=bs[v], mode="lines+markers", name=VIEW_LABEL[v],
                            line=dict(width=2, color=VIEW_COLORS[v]), marker=dict(size=8),
                            hovertemplate=f"{VIEW_LABEL[v]}<br>%{{x}}: RPS %{{y:.4f}}<extra></extra>")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="RPS (lower is better)",
                          legend=dict(orientation="h", y=1.15), hovermode="x unified")
        st.markdown("##### RPS by season")
        st.plotly_chart(fig, use_container_width=True)
    o = pipeline.outputs(code)
    if o["backtest"].exists():
        bt = load_parquet(str(o["backtest"]), mtime(o["backtest"]))
        cal = metrics.calibration(metrics.probs_of(bt, "ens"), bt["y"].to_numpy(int))
        cal = cal[cal["n"] >= 30]  # hide bins too thin to say anything
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(color="#9a9890", dash="dot"))
        fig.add_scatter(x=cal["predicted"], y=cal["observed"], mode="lines+markers", name="Ensemble",
                        line=dict(color=VIEW_COLORS["ens"], width=2), marker=dict(size=8),
                        customdata=cal["n"], hovertemplate="said %{x:.0%} → happened %{y:.0%} (n=%{customdata})<extra></extra>")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Predicted", tickformat=".0%"),
                          yaxis=dict(title="Actually happened", tickformat=".0%"), legend=dict(orientation="h", y=1.1))
        a, b = st.columns(2)
        with a:
            st.markdown("##### Calibration: when it says 60%, does it happen 60%?")
            st.plotly_chart(fig, use_container_width=True)
        if bt["odds_h_open"].notna().any():
            with b:
                st.markdown("##### If you had followed each view (flat 1-unit value bets)")
                me = st.slider("Minimum edge", 0.0, 0.20, 0.05, 0.01)
                curves = betting.equity_curves(bt, ["dc", "gbm", "mkt", "ens"], min_edge=me)
                if not curves.empty:
                    fig2 = go.Figure()
                    for v, g in curves.groupby("view"):
                        fig2.add_scatter(x=g["bet_no"], y=g["profit_units"], mode="lines", name=VIEW_LABEL[v],
                                         line=dict(width=2, color=VIEW_COLORS[v]))
                    fig2.add_hline(y=0, line_color="#9a9890", line_width=1)
                    fig2.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                                       xaxis_title="Bet number", yaxis_title="Profit (units)",
                                       legend=dict(orientation="h", y=1.1))
                    st.plotly_chart(fig2, use_container_width=True)
            sim = betting.simulate_all(bt, ["dc", "gbm", "mkt", "ens"], min_edge=me)
            if not sim.empty:
                sim.index = sim.index.map(VIEW_LABEL)
                st.dataframe(sim[["bets", "hit_rate", "avg_odds", "roi", "roi_ci_low", "roi_ci_high", "clv",
                                  "kelly_max_drawdown"]].style.format(
                    {"hit_rate": "{:.1%}", "avg_odds": "{:.2f}", "roi": "{:+.1%}", "roi_ci_low": "{:+.1%}",
                     "roi_ci_high": "{:+.1%}", "clv": "{:+.1%}", "kelly_max_drawdown": "{:.0%}"}),
                    use_container_width=True)
                st.caption("ROI interval = 95% bootstrap · CLV = vs. closing line")
    with st.expander("Raw track record JSON"):
        st.json(json.loads(json.dumps(tr, default=str)))


def page_journal(code: str, matches: pd.DataFrame):
    if st.button("Settle finished bets for this league"):
        n = journal.settle(matches, code)
        st.success(f"Settled {n} bets.")
    bets = journal.load()
    if bets.empty:
        st.info("No bets logged yet. Use 'Log a bet' in any match report.")
        return
    summ = journal.summary(bets)
    if not summ.empty:
        st.markdown("##### Results by the view you followed")
        st.dataframe(summ.style.format({"staked": "{:.2f}", "profit": "{:+.2f}", "hit_rate": "{:.1%}",
                                        "avg_odds": "{:.2f}", "avg_clv": "{:+.1%}", "roi": "{:+.1%}"}),
                     use_container_width=True)
    st.markdown("##### All bets")
    st.dataframe(bets, hide_index=True, use_container_width=True)
    with st.expander("Delete a bet"):
        bid = st.number_input("Bet id", min_value=1, step=1)
        if st.button("Delete"):
            journal.delete(int(bid))
            st.rerun()


def page_help():
    st.markdown("""
### How to read this

**The views** - each is an independent opinion about the match:

| View | What it is | Sees odds? |
|---|---|---|
| Elo rating | Long-run team strength, updated after every match | No |
| Goals model (Dixon-Coles) | Estimates each team's attack & defence, predicts the score | No |
| Stats model | Machine learning on form, shots, rest, ratings, head-to-head | No |
| Stats + market | Same, but also sees the opening odds | **Yes** |
| Ensemble | Weighted blend of Elo, goals and stats models | No |
| Bookmakers (fair) | The odds with the bookmaker's margin removed | It *is* the odds |

**Agreement** - when the views agree the picture is clear; when they disagree, the match is
genuinely uncertain (or one view knows something the others don't).

**Edge** = model probability × odds − 1. Positive means the price is higher than the model's fair price.
Bookmakers are very good: most "edges" under 5% are noise. The most trustworthy evidence that a view has a
real edge is **CLV** (beating the closing line) in the Track record page.

**Fair odds** = 1 / probability. **¼-Kelly stake** = a conservative stake size relative to your bankroll,
capped at 2%.

**RPS** (ranked probability score) is the standard accuracy metric for football forecasts - lower is better.
Typical top-league values are ~0.19-0.21.

This tool supports decisions - it does not place bets, and no model makes football predictable.
Only bet what you can afford to lose.
""")


# ----------------------------------------------------------------------------- main
def main():
    st.title("⚽ Ideamillonaria")
    trained = trained_leagues()
    if not trained:
        st.warning("No trained leagues yet. In a terminal run for example:\n\n"
                   "`ideam update CHI1 EPL INT`  (or `ideam demo` for an offline demo)")
        page_help()
        return
    with st.sidebar:
        code = st.selectbox("League", trained, format_func=lambda c: f"{c} - {get_league(c).country} {get_league(c).name}")
        st.session_state["bankroll"] = st.number_input("Bankroll (for stake suggestions)", 1.0, 1e9, 100.0, 10.0)
        st.caption("Data refresh: `ideam update " + code + "` in a terminal.")
    art = load_art(code, mtime(pipeline.artifacts_path(code)))
    matches = load_matches(code, mtime(store.matches_path(code)))
    tabs = st.tabs(["Upcoming", "Custom matchup", "Track record", "Bet journal", "How to read this"])
    with tabs[0]:
        page_upcoming(code, art, matches)
    with tabs[1]:
        page_custom(code, art, matches)
    with tabs[2]:
        page_track(code, art)
    with tabs[3]:
        page_journal(code, matches)
    with tabs[4]:
        page_help()


main()

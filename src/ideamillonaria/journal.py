"""Personal bet journal (SQLite). Record what you actually bet and which view
suggested it; settle automatically from results; see which views really helped."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime

import numpy as np
import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    placed_at TEXT NOT NULL,
    league TEXT NOT NULL,
    match_date TEXT NOT NULL,
    home TEXT NOT NULL,
    away TEXT NOT NULL,
    market TEXT NOT NULL,          -- 1X2 | OU25
    selection TEXT NOT NULL,       -- h | d | a | over | under
    odds REAL NOT NULL,
    stake REAL NOT NULL,
    view TEXT,                     -- which model view you followed (ens, gbm, mkt, dc, own)
    model_prob REAL,
    closing_odds REAL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | won | lost | void
    profit REAL,
    note TEXT
)
"""


def _conn(path=None):
    config.ensure_dirs()
    con = sqlite3.connect(path or config.JOURNAL_PATH)
    con.execute(SCHEMA)
    return con


def add_bet(league: str, match_date, home: str, away: str, market: str, selection: str, odds: float,
            stake: float, view: str = "own", model_prob: float | None = None, note: str = "",
            path=None) -> int:
    with closing(_conn(path)) as con, con:
        cur = con.execute(
            "INSERT INTO bets (placed_at, league, match_date, home, away, market, selection, odds, stake, view,"
            " model_prob, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), league, str(pd.Timestamp(match_date).date()), home,
             away, market, selection, float(odds), float(stake), view, model_prob, note),
        )
        return int(cur.lastrowid)


def load(path=None) -> pd.DataFrame:
    with closing(_conn(path)) as con:
        return pd.read_sql_query("SELECT * FROM bets ORDER BY match_date, id", con)


def delete(bet_id: int, path=None) -> None:
    with closing(_conn(path)) as con, con:
        con.execute("DELETE FROM bets WHERE id = ?", (int(bet_id),))


def _won(selection: str, market: str, hg: float, ag: float) -> bool:
    if market == "OU25":
        return (hg + ag > 2.5) if selection == "over" else (hg + ag < 2.5)
    res = "h" if hg > ag else ("d" if hg == ag else "a")
    return res == selection


def settle(matches: pd.DataFrame, league: str, path=None) -> int:
    """Settle pending bets of ``league`` whose match result is now known."""
    bets = load(path)
    pend = bets[(bets["status"] == "pending") & (bets["league"] == league)]
    if pend.empty:
        return 0
    played = matches[matches["hg"].notna()]
    n = 0
    with closing(_conn(path)) as con, con:
        for _, b in pend.iterrows():
            d = pd.Timestamp(b["match_date"])
            m = played[(played["home"] == b["home"]) & (played["away"] == b["away"])
                       & ((played["date"] - d).abs() <= pd.Timedelta(days=2))]
            if m.empty:
                continue
            r = m.iloc[0]
            won = _won(b["selection"], b["market"], r["hg"], r["ag"])
            profit = b["stake"] * (b["odds"] - 1) if won else -b["stake"]
            close = None
            if b["market"] == "1X2":
                close = r.get(f"odds_{b['selection']}_close")
            con.execute("UPDATE bets SET status=?, profit=?, closing_odds=COALESCE(closing_odds, ?) WHERE id=?",
                        ("won" if won else "lost", float(profit),
                         None if close is None or pd.isna(close) else float(close), int(b["id"])))
            n += 1
    return n


def summary(bets: pd.DataFrame) -> pd.DataFrame:
    s = bets[bets["status"].isin(["won", "lost"])]
    if s.empty:
        return pd.DataFrame()
    s = s.assign(clv=np.where(s["closing_odds"].notna(), s["odds"] / s["closing_odds"] - 1, np.nan))
    g = s.groupby("view").agg(bets=("id", "size"), staked=("stake", "sum"), profit=("profit", "sum"),
                              hit_rate=("status", lambda x: (x == "won").mean()), avg_odds=("odds", "mean"),
                              avg_clv=("clv", "mean"))
    g["roi"] = g["profit"] / g["staked"]
    total = pd.DataFrame([{"bets": len(s), "staked": s["stake"].sum(), "profit": s["profit"].sum(),
                           "hit_rate": (s["status"] == "won").mean(), "avg_odds": s["odds"].mean(),
                           "avg_clv": s["clv"].mean(), "roi": s["profit"].sum() / s["stake"].sum()}],
                         index=pd.Index(["ALL"], name="view"))
    return pd.concat([g, total])

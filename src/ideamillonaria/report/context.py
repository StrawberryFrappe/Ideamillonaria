"""Context tables for a match report: recent form, head-to-head, league table."""

from __future__ import annotations

import numpy as np
import pandas as pd


def team_matches(matches: pd.DataFrame, team: str, before=None, n: int = 10) -> pd.DataFrame:
    m = matches[(matches["hg"].notna()) & ((matches["home"] == team) | (matches["away"] == team))]
    if before is not None:
        m = m[m["date"] < pd.Timestamp(before)]
    m = m.sort_values("date").tail(n)
    is_home = m["home"] == team
    gf = np.where(is_home, m["hg"], m["ag"])
    ga = np.where(is_home, m["ag"], m["hg"])
    return pd.DataFrame({
        "date": m["date"].dt.date,
        "venue": np.where(m["neutral"], "N", np.where(is_home, "H", "A")),
        "opponent": np.where(is_home, m["away"], m["home"]),
        "score": [f"{int(a)}-{int(b)}" for a, b in zip(gf, ga)],
        "result": np.where(gf > ga, "W", np.where(gf == ga, "D", "L")),
        "shots": np.where(is_home, m["h_shots"], m["a_shots"]),
        "shots_against": np.where(is_home, m["a_shots"], m["h_shots"]),
    }).iloc[::-1].reset_index(drop=True)


def form_string(matches: pd.DataFrame, team: str, before=None, n: int = 5) -> str:
    t = team_matches(matches, team, before, n)
    return "".join(t["result"].tolist())  # most recent first


def head_to_head(matches: pd.DataFrame, home: str, away: str, before=None, n: int = 10) -> pd.DataFrame:
    m = matches[matches["hg"].notna() & (
        ((matches["home"] == home) & (matches["away"] == away))
        | ((matches["home"] == away) & (matches["away"] == home)))]
    if before is not None:
        m = m[m["date"] < pd.Timestamp(before)]
    m = m.sort_values("date", ascending=False).head(n)
    return pd.DataFrame({
        "date": m["date"].dt.date, "home": m["home"], "away": m["away"],
        "score": [f"{int(a)}-{int(b)}" for a, b in zip(m["hg"], m["ag"])],
        "competition": m["tournament"].fillna(m["league"]),
    }).reset_index(drop=True)


def standings(matches: pd.DataFrame, season: str, before=None) -> pd.DataFrame:
    m = matches[(matches["season"] == season) & matches["hg"].notna()]
    if before is not None:
        m = m[m["date"] < pd.Timestamp(before)]
    if m.empty:
        return pd.DataFrame()
    rows = []
    for side, gf, ga in (("home", "hg", "ag"), ("away", "ag", "hg")):
        rows.append(pd.DataFrame({"team": m[side], "gf": m[gf], "ga": m[ga], "date": m["date"]}))
    long = pd.concat(rows).sort_values("date")
    long["w"] = (long["gf"] > long["ga"]).astype(int)
    long["d"] = (long["gf"] == long["ga"]).astype(int)
    long["l"] = (long["gf"] < long["ga"]).astype(int)
    long["res"] = np.where(long["w"] == 1, "W", np.where(long["d"] == 1, "D", "L"))
    t = long.groupby("team").agg(P=("w", "size"), W=("w", "sum"), D=("d", "sum"), L=("l", "sum"),
                                 GF=("gf", "sum"), GA=("ga", "sum"),
                                 Form=("res", lambda s: "".join(s.tail(5).tolist()[::-1])))
    t["GD"] = t["GF"] - t["GA"]
    t["Pts"] = 3 * t["W"] + t["D"]
    t = t.sort_values(["Pts", "GD", "GF"], ascending=False)
    t.insert(0, "Pos", range(1, len(t) + 1))
    return t[["Pos", "P", "W", "D", "L", "GF", "GA", "GD", "Pts", "Form"]].astype(
        {c: int for c in ["P", "W", "D", "L", "GF", "GA", "GD", "Pts"]})


def rest_days(matches: pd.DataFrame, team: str, date) -> float:
    m = matches[(matches["hg"].notna()) & ((matches["home"] == team) | (matches["away"] == team))
                & (matches["date"] < pd.Timestamp(date))]
    if m.empty:
        return np.nan
    return float((pd.Timestamp(date) - m["date"].max()).days)

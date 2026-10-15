"""Turn model internals into plain language for someone who doesn't follow football."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..evaluate.betting import edge, fair_odds

OUTCOME_NAMES = ["home win", "draw", "away win"]

_CORE = {
    "form_pts": ("points per game, last {n}", "{:.2f}"),
    "form_gf": ("goals scored per game, last {n}", "{:.2f}"),
    "form_ga": ("goals conceded per game, last {n}", "{:.2f}"),
    "form_shots_f": ("shots per game, last {n}", "{:.1f}"),
    "form_shots_a": ("shots allowed per game, last {n}", "{:.1f}"),
    "form_sot_f": ("shots on target per game, last {n}", "{:.1f}"),
    "form_sot_a": ("shots on target allowed per game, last {n}", "{:.1f}"),
    "form_surprise": ("results vs. what ratings expected, last {n} (+ = overperforming)", "{:+.2f}"),
    "venue_pts": ("points per game in last {n} games at this venue type", "{:.2f}"),
    "venue_gf": ("goals scored per game in last {n} games at this venue type", "{:.2f}"),
    "venue_ga": ("goals conceded per game in last {n} games at this venue type", "{:.2f}"),
    "season_ppg": ("points per game this season", "{:.2f}"),
    "season_gd_pg": ("goal difference per game this season", "{:+.2f}"),
    "season_played": ("games played this season", "{:.0f}"),
    "rest_days": ("days of rest since previous match", "{:.0f}"),
    "matches_played": ("matches in the database", "{:.0f}"),
}


def describe(feature: str, value: float, home: str, away: str) -> str:
    v = value
    if feature == "elo_diff":
        fav = home if v >= 0 else away
        return f"Strength gap (Elo, incl. home advantage): {abs(v):.0f} points in favour of {fav}"
    if feature == "elo_h":
        return f"{home} Elo rating: {v:.0f}"
    if feature == "elo_a":
        return f"{away} Elo rating: {v:.0f}"
    if feature == "h2h_gd_5":
        return f"Head-to-head: {home} averaged {v:+.1f} goal difference in recent meetings"
    if feature == "h2h_n":
        return f"{v:.0f} previous meetings between these teams"
    if feature == "neutral":
        return "Played on neutral ground" if v else "Normal home/away match"
    if feature == "importance":
        return f"Match importance level {v:.1f} (0 friendly - 3 World Cup)"
    if feature == "dc_lambda_h":
        return f"Goals model expects {home} to score {v:.2f}"
    if feature == "dc_lambda_a":
        return f"Goals model expects {away} to score {v:.2f}"
    if feature.startswith("p_dc_"):
        what = {"h": f"{home} win", "d": "a draw", "a": f"{away} win", "o25": "over 2.5 goals"}[feature[5:]]
        return f"Goals model gives {what} {v:.0%}"
    m = re.match(r"^(diff_|h_|a_)(.+?)(?:_(\d+))?$", feature)
    if m:
        prefix, core, n = m.groups()
        label, fmt = _CORE.get(core, (core.replace("_", " "), "{:.2f}"))
        label = label.format(n=n or "")
        if prefix == "diff_":
            return f"Gap in {label} ({home} minus {away}): {v:+.2f}"
        team = home if prefix == "h_" else away
        try:
            return f"{team}: {fmt.format(v)} {label}"
        except (ValueError, TypeError):
            return f"{team}: {label} n/a"
    return f"{feature.replace('_', ' ')}: {v:.2f}"


def reasons(contrib: np.ndarray, features: list[str], x: pd.Series, outcome_idx: int, home: str, away: str,
            top: int = 5) -> pd.DataFrame:
    """Top features pushing the stats model towards / away from one outcome.

    ``contrib``: (3, n_features + 1) SHAP values for a single match.
    """
    c = contrib[outcome_idx, :-1]
    order = np.argsort(-np.abs(c))
    rows = []
    for i in order:
        if len(rows) >= top:
            break
        f = features[i]
        val = x[f]
        if pd.isna(val) or abs(c[i]) < 1e-4:
            continue
        rows.append({
            "reason": describe(f, float(val), home, away),
            "effect": "for" if c[i] > 0 else "against",
            "strength": float(c[i]),
        })
    return pd.DataFrame(rows)


def best_value(row: pd.Series, views=("ens", "gbm", "mkt", "dc"), when: str = "open",
               min_edge: float = 0.03) -> list[dict]:
    """Edges of each view against the available bookmaker odds for 1X2."""
    out = []
    odds = np.array([row.get(f"odds_{o}_{when}", np.nan) for o in "hda"], float)
    if np.isnan(odds).all():
        return out
    for v in views:
        p = np.array([row.get(f"p_{v}_{o}", np.nan) for o in "hda"], float)
        if np.isnan(p).any():
            continue
        e = edge(p, odds)
        for k in range(3):
            if np.isfinite(e[k]) and e[k] >= min_edge:
                out.append({"view": v, "outcome": k, "probability": p[k], "odds": odds[k],
                            "fair_odds": float(fair_odds(p[k])), "edge": float(e[k])})
    return sorted(out, key=lambda d: -d["edge"])


def summary(row: pd.Series, home_form: str = "", away_form: str = "") -> list[str]:
    """A handful of plain-language bullet points about a match panel row."""
    home, away = row["home"], row["away"]
    names = [f"{home} win", "draw", f"{away} win"]
    ens = np.array([row["p_ens_h"], row["p_ens_d"], row["p_ens_a"]])
    k = int(np.nanargmax(ens))
    lines = [f"**Most likely: {names[k]} ({ens[k]:.0%})**. Full split: {home} {ens[0]:.0%} / "
             f"draw {ens[1]:.0%} / {away} {ens[2]:.0%}."]
    if ens[k] < 0.45:
        lines.append("No outcome is a clear favourite - this is a genuinely open match.")
    if pd.notna(row.get("dc_lambda_h")):
        lines.append(f"Goals model expects roughly **{row['dc_lambda_h']:.1f} - {row['dc_lambda_a']:.1f}** "
                     f"(over 2.5 goals: {row['p_dc_o25']:.0%}).")
    if home_form or away_form:
        lines.append(f"Recent form (latest first): {home} `{home_form or '-'}`, {away} `{away_form or '-'}`.")
    views = [v for v in ("elo", "dc", "gbm", "mkt", "book_open") if pd.notna(row.get(f"p_{v}_h"))]
    ph = [row[f"p_{v}_h"] for v in views]
    if ph:
        lines.append(f"Agreement between views: **{row['agreement']}** - {home} win ranges "
                     f"{min(ph):.0%}-{max(ph):.0%} across {len(views)} views.")
    if pd.notna(row.get("p_book_open_h")):
        bk = np.array([row["p_book_open_h"], row["p_book_open_d"], row["p_book_open_a"]])
        diff = ens - bk
        j = int(np.argmax(np.abs(diff)))
        lines.append(f"Bookmakers (margin removed) say {home} {bk[0]:.0%} / draw {bk[1]:.0%} / {away} {bk[2]:.0%}. "
                     f"Biggest disagreement: {names[j]} ({diff[j]:+.0%} models vs market).")
        val = best_value(row)
        ens_val = [v for v in val if v["view"] == "ens"]
        if val:
            b = ens_val[0] if ens_val else val[0]
            n_views = len({v["view"] for v in val if v["outcome"] == b["outcome"]})
            lines.append(f"Possible value: **{names[b['outcome']]}** at {b['odds']:.2f} - the {b['view']} view "
                         f"puts fair odds at {b['fair_odds']:.2f} (edge {b['edge']:+.1%}); {n_views} of 4 views "
                         f"see value here. Markets are usually right: one model disagreeing is weak evidence, "
                         f"several agreeing is stronger. Treat edges under ~5% as noise.")
        else:
            lines.append("No meaningful value vs the current odds: models and market roughly agree.")
    else:
        lines.append("No bookmaker odds stored for this match - enter odds in the report to check for value.")
    return lines

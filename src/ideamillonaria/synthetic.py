"""Synthetic league generator with known ground truth.

Used by the tests and by ``ideam demo`` (so the full pipeline, including odds and the
betting layer, can be exercised without network access). Team strengths drift over
time, three teams are relegated/promoted each season, and a simulated bookmaker
prices every match from noisy true probabilities plus a margin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .data.schema import conform


def _probs(lam, mu, max_goals=10):
    g = np.arange(max_goals + 1)
    p = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    return np.array([np.tril(p, -1).sum(), np.trace(p), np.triu(p, 1).sum()]), p


def generate_league(code: str = "SYN", n_teams: int = 18, seasons: int = 8, start_year: int = 2015,
                    home_adv: float = 0.25, seed: int = 7, margin: float = 0.06,
                    with_fixtures: int = 9) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    pool = [f"Team {chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(n_teams + 3 * seasons + 3)]
    active = pool[:n_teams]
    next_new = n_teams
    att = {t: rng.normal(0, 0.3) for t in pool}
    dfn = {t: rng.normal(0, 0.3) for t in pool}
    for t in pool[n_teams:]:  # promoted teams are weaker on average
        att[t] -= 0.2
        dfn[t] -= 0.2
    rows = []
    for s in range(seasons):
        year = start_year + s
        day0 = pd.Timestamp(f"{year}-08-10")
        fixtures = [(h, a) for h in active for a in active if h != a]
        rng.shuffle(fixtures)
        # greedy rounds: every team plays at most once per matchday
        rounds: list[list[tuple[str, str]]] = []
        for h, a in fixtures:
            for rd in rounds:
                if all(h not in pair and a not in pair for pair in rd):
                    rd.append((h, a))
                    break
            else:
                rounds.append([(h, a)])
        points = {t: 0 for t in active}
        for r, rd in enumerate(rounds):
            date = day0 + pd.Timedelta(days=4 * r)
            for h, a in rd:
                lam = np.exp(0.15 + home_adv + att[h] - dfn[a])
                mu = np.exp(0.15 + att[a] - dfn[h])
                hg, ag = rng.poisson(lam), rng.poisson(mu)
                p, mat = _probs(lam, mu)
                # bookmaker: noisy view of the truth, closing line sharper than opening
                p_open = np.clip(p * np.exp(rng.normal(0, 0.10, 3)), 0.01, None)
                p_open /= p_open.sum()
                p_close = np.clip(p * np.exp(rng.normal(0, 0.04, 3)), 0.01, None)
                p_close /= p_close.sum()
                over = mat[np.add.outer(np.arange(11), np.arange(11)) > 2].sum()
                rows.append({
                    "league": code, "date": date, "home": h, "away": a, "hg": hg, "ag": ag,
                    "h_shots": rng.poisson(lam * 8 + 3), "a_shots": rng.poisson(mu * 8 + 3),
                    "h_sot": rng.poisson(lam * 3 + 1), "a_sot": rng.poisson(mu * 3 + 1),
                    "odds_h_open": 1 / (p_open[0] * (1 + margin)), "odds_d_open": 1 / (p_open[1] * (1 + margin)),
                    "odds_a_open": 1 / (p_open[2] * (1 + margin)),
                    "odds_h_close": 1 / (p_close[0] * (1 + margin)), "odds_d_close": 1 / (p_close[1] * (1 + margin)),
                    "odds_a_close": 1 / (p_close[2] * (1 + margin)),
                    "odds_o25_open": 1 / (over * (1 + margin)), "odds_u25_open": 1 / ((1 - over) * (1 + margin)),
                    "odds_o25_close": 1 / (over * (1 + margin)), "odds_u25_close": 1 / ((1 - over) * (1 + margin)),
                    "true_p_h": p[0], "true_p_d": p[1], "true_p_a": p[2],
                })
                points[h] += 3 if hg > ag else (1 if hg == ag else 0)
                points[a] += 3 if ag > hg else (1 if hg == ag else 0)
            for t in active:  # weekly random-walk drift of strengths
                att[t] += rng.normal(0, 0.012)
                dfn[t] += rng.normal(0, 0.012)
        # relegation / promotion
        table = sorted(active, key=lambda t: points[t])
        for t in table[:3]:
            active.remove(t)
            active.append(pool[next_new])
            next_new += 1
    df = pd.DataFrame(rows)
    truth = {"att": att, "dfn": dfn, "home_adv": home_adv}
    if with_fixtures:
        last = df["date"].max() + pd.Timedelta(days=7)
        sched = []
        for i in range(0, min(with_fixtures * 2, len(active)), 2):
            h, a = active[i], active[i + 1]
            p, mat = _probs(np.exp(0.15 + home_adv + att[h] - dfn[a]), np.exp(0.15 + att[a] - dfn[h]))
            p_open = np.clip(p * np.exp(rng.normal(0, 0.10, 3)), 0.01, None)
            p_open /= p_open.sum()
            over = mat[np.add.outer(np.arange(11), np.arange(11)) > 2].sum()
            sched.append({"league": code, "date": last, "home": h, "away": a,
                          "odds_h_open": 1 / (p_open[0] * (1 + margin)), "odds_d_open": 1 / (p_open[1] * (1 + margin)),
                          "odds_a_open": 1 / (p_open[2] * (1 + margin)),
                          "odds_o25_open": 1 / (over * (1 + margin)), "odds_u25_open": 1 / ((1 - over) * (1 + margin))})
        df = pd.concat([df, pd.DataFrame(sched)], ignore_index=True)
    df["season"] = df["date"].apply(lambda d: f"{d.year if d.month >= 7 else d.year - 1}-"
                                              f"{str((d.year if d.month >= 7 else d.year - 1) + 1)[-2:]}")
    true_probs = df[["date", "home", "away", "true_p_h", "true_p_d", "true_p_a"]].dropna()
    return conform(df), {**truth, "true_probs": true_probs}

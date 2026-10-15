# Ideamillonaria ⚽

A football (soccer) prediction and **decision-support** tool. For every match it shows several independent
statistical views side by side, shows the key factors behind each pick, prices every common betting
market, and keeps an honest out-of-sample track record of each view.

It **does not place bets**. It is there to help you make the most informed decision you can.

Runs comfortably on any laptop (a GTX 1650 is more than enough; everything runs on CPU in seconds to a
couple of minutes).

## What it covers

| Code | League | Data source (automatic, no manual entry) |
|---|---|---|
| `CHI1`, `CHI2` | Chile Primera División, Primera B | ESPN public API (results, fixtures, shots, possession, odds when listed). Fallback: football-data.org |
| `EPL`, `ELC`, `LALIGA`, `LALIGA2`, `SERIEA`, `SERIEB`, `BUNDESLIGA`, `BUNDESLIGA2`, `LIGUE1`, `LIGUE2`, `EREDIVISIE`, `PRIMEIRA` | Top European leagues | football-data.co.uk (results, shots, corners, opening + closing odds since ~2005) |
| `ARG`, `BRA`, `MEX` | Argentina, Brazil, Mexico | football-data.co.uk (results + closing odds) |
| `INT` | All national-team matches since 1872 | martj42/international_results |
| `SYN` | Synthetic league with known truth | Generated locally (offline demo) |

Odds for Chile: no free source of *historical* Chilean odds exists. If you add a free
[The Odds API](https://the-odds-api.com) key, `ideam update` saves a snapshot of current prices every time it
runs. Over time that builds an odds history, so Chile also gets a betting backtest.

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env             # optional: add API keys
```

## Use

```bash
ideam doctor                     # which data sources can this machine reach?
ideam demo                       # offline demo on a synthetic league (1 minute)

ideam update CHI1 EPL INT        # fetch -> snapshot odds -> backtest + train -> predict -> settle journal
ideam backtest EPL               # accuracy vs bookmakers + value-bet backtest per view
ideam predict CHI1               # upcoming fixtures
ideam matchup INT Chile Peru --neutral
ideam dashboard                  # the web app (http://localhost:8501)
```

`ideam update` is the only command you need day to day. You can schedule it, for example with Windows Task
Scheduler or cron, once a day.

## The views (side by side for every match)

| View | What it is | Sees odds? |
|---|---|---|
| **Elo** | Long-run team strength, updated after every match (home advantage, goal margin, season regression, promoted teams start low) | no |
| **Goals model** (Dixon-Coles) | Each team's attack and defence strength with time decay. It predicts the full scoreline grid, which gives 1X2, over/under, BTTS, handicaps, exact scores and team totals | no |
| **Stats model** (LightGBM) | Machine learning on ~70 pre-match features: form (last 5/10), home/away form, shots and shots on target, results vs expectation, season points, rest days, head-to-head, Elo, goals model | no |
| **Stats + market** | The same model, plus the bookmakers' opening odds as features. A separate model, never mixed into the pure ones | yes |
| **Ensemble** | A blend of Elo, goals and stats models. Its weights are fit on earlier seasons' out-of-sample log-loss | no |
| **Bookmakers** | Opening and closing odds with the margin removed (power method) | it *is* the odds |

Each view is a separate, honest opinion. The **agreement meter** shows whether they line up. When the pure
models and the market disagree, that is where the information is, but remember the market is usually right.

### How betting is kept from skewing the statistics

- The pure models never see odds. A test (`tests/test_leakage.py`) checks this.
- Models and hyperparameters are compared only on proper scoring rules (RPS, log-loss, Brier). They are
  never compared on betting profit, because profit over a few hundred bets is mostly noise.
- Everything is **walk-forward**: each season is predicted by models trained only on earlier seasons.
  Dixon-Coles is refit weekly on past matches only. A test checks that deleting the future does not change
  any feature.
- The betting layer (`evaluate/betting.py`) is read-only. It reports ROI with a 95% bootstrap interval and
  **closing-line value** (CLV), which is the most reliable sign of a real edge.

## The dashboard

- **Upcoming**: fixtures with the ensemble's probabilities, pick, agreement and value flags.
- **Match report**:
  - A one-line header (pick, expected goals, likely score, agreement, value) and every view side by side,
    with each view's recent track record. Explanations are in hover tooltips.
  - Odds input showing your edge, fair odds and a ¼-Kelly stake suggestion (capped at 2% of bankroll).
  - A scoreline heatmap and all markets with fair odds.
  - *Why?*: the stats model's top reasons, from SHAP values.
  - Recent form, rest days, head-to-head, the league table, and ESPN lineups/injuries for Chile.
- **Custom matchup**: any two teams on any date. This is how national-team and World Cup games are
  predicted.
- **Track record**: accuracy per view, comparison with the bookmakers, RPS by season, a calibration plot,
  and equity curves from following each view.
- **Bet journal**: log the bets you actually place. They settle automatically, and you see profit, hit rate
  and CLV per view you followed.

## Measured results

Out-of-sample on **national-team matches 2014-2026** (12,369 matches, every view scored on the same matches):

| View | RPS ↓ | Log-loss ↓ | Accuracy |
|---|---|---|---|
| Base rates (know nothing) | 0.2286 | 1.052 | 47.6% |
| Elo | 0.1730 | 0.881 | 59.8% |
| Goals model | 0.1716 | 0.875 | 60.3% |
| Stats model | 0.1705 | 0.871 | 60.2% |
| **Ensemble** | **0.1699** | **0.868** | 60.2% |

Club leagues are harder: around 0.19-0.21 RPS and 50-55% accuracy is typical for top leagues. Bookmakers'
closing prices are the toughest benchmark in the sport. Run `ideam backtest EPL` on your machine to see
exactly where this model stands against them.

## Honest caveats

- No model makes football predictable. Draws are close to unpredictable, and single matches are noisy.
- If the backtest shows a positive ROI whose 95% interval includes 0, it is indistinguishable from luck.
- ESPN's API is undocumented and could change. `ideam doctor` tells you, and football-data.org is the
  fallback (free key).
- Bet only what you can afford to lose.

## Project layout

```
src/ideamillonaria/
  config.py              leagues, model parameters, paths
  data/                  canonical schema, source adapters, storage, team-name matching
  models/                elo, dixon_coles, features, gbm, ensemble, market
  evaluate/              walk-forward backtest, scoring rules, betting simulation
  report/                markets from the score grid, match headline + key factors, context tables
  pipeline.py            train -> backtest -> final fit -> predict
  journal.py             personal bet log (SQLite)
  cli.py                 `ideam` command
app/dashboard.py         Streamlit web app
tests/                   pytest (leakage, purity, parsers, models, betting, end-to-end)
```

Data, trained models and your journal live in `data/` (git-ignored).

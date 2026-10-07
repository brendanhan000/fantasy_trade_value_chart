"""Estimate config values from data (optional; prints suggestions, never edits config).

  python calibrate.py --mode forecast                 # recency_decay / prior_games / floor by
                                                      #   one-game-ahead forecast error (last season)
  python calibrate.py --mode injuries                 # P(plays | injury-report status), 2021-2025
  python calibrate.py --mode backtest --weeks 4 6 8 10   # b1, lambda, g1, g2, k vs realized ROS VORP
  python calibrate.py --mode market   --weeks 5       # same params vs FantasyCalc
  python calibrate.py --mode benchmark --season 2024 2025   # model vs plain-PPG ranking accuracy

backtest/market compare on a 0-100 scale with alpha = 1 (model only), so the
market isn't fitted to itself. Backtest pools several weeks: one week of one
season is too few outcomes to pin down five parameters.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import logging

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
from scipy.stats import spearmanr

from main import ROOT, load_config
from data import nflverse as nv
from model import scoring as sc
from model.features import load_inputs, prior_games
from model.layers.l3_vorp import replacement_levels
from model.pipeline import player_pool, run_model

PARAMS = [("layer1", "b1", 0.0, 1.0), ("layer2", "lambda", 0.0, 0.5),
          ("layer2", "g1", 0.0, 15.0), ("layer2", "g2", 0.0, 15.0), ("layer6", "k", 1.0, 2.0)]


def apply(cfg: dict, x: np.ndarray) -> dict:
    c = copy.deepcopy(cfg)
    for (sec, key, lo, hi), v in zip(PARAMS, x):
        c[sec][key] = float(np.clip(v, lo, hi))
    c["layer1"]["b2"] = 1 - c["layer1"]["b1"]
    c["layer6"]["alpha"] = 1.0
    return c


def season_points(season: int, cfg: dict) -> pd.DataFrame:
    s = nv.player_stats(season)
    s = s.merge(nv.pbp_player_week(season), on=["season", "week", "gsis_id"], how="left")
    s["pts"] = (sc.base_points(s, cfg["scoring"]) + sc.milestone_points(s, cfg["scoring"])
                + sc.long_td_points(s, cfg["scoring"]))
    return s


def realized_vorp(season: int, week: int, cfg: dict) -> pd.Series:
    """Actual rest-of-season points over replacement, same replacement rule as the model."""
    s = season_points(season, cfg)
    s = s[s.week.between(week, cfg["league"]["last_week"])]
    omega = {int(w): v for w, v in cfg["layer3"]["week_weights"].items()}
    parts = []
    for w, g in s.groupby("week"):
        g = g.set_index("gsis_id")
        R = replacement_levels(g.pts, g.position, cfg)
        parts.append((g.pts - g.position.map(R)) * omega.get(int(w), 1.0))
    return pd.concat(parts).groupby(level=0).sum()


def fit_forecast(cfg: dict, season: int) -> None:
    """Grid-search the projection weights by one-game-ahead MSE on fantasy points.

    Exponential weighting is the optimal (Kalman steady-state) forecast for a
    random-walk-plus-noise talent model; its decay is fitted by out-of-sample
    forecast error, the standard way exponential-smoothing parameters are set.
    """
    prev, cur = season_points(season - 1, cfg), season_points(season, cfg)
    n_prev = prev.groupby("gsis_id").size()
    prev_mean = prev.groupby("gsis_id").pts.mean()
    pos = pd.concat([prev, cur]).groupby("gsis_id").position.last()
    weeks = range(2, cfg["league"]["last_week"] + 1)

    def mse(decay: float, pg0: float, floor: float) -> float:
        c = copy.deepcopy(cfg)
        c["projection"].update(prior_games=pg0, prior_games_floor=floor)
        err, n = 0.0, 0
        for w in weeks:
            h = cur[cur.week < w]
            wt = decay ** (w - 1 - h.week)
            cw = wt.groupby(h.gsis_id).sum()
            cx = (wt * h.pts).groupby(h.gsis_id).sum()
            pw = np.minimum(1.0, prior_games(c, w) / n_prev)  # weight per 2025 game
            ids = cw.index.union(pw.index)
            pwt = (pw * n_prev).reindex(ids, fill_value=0)
            num = cx.reindex(ids, fill_value=0) + (pwt * prev_mean.reindex(ids)).fillna(0)
            den = cw.reindex(ids, fill_value=0) + pwt
            xbar = num / den
            games = h.groupby("gsis_id").size().reindex(ids, fill_value=0) + n_prev.reindex(ids, fill_value=0)
            base = xbar[games >= 4].groupby(pos).median().reindex(pos.reindex(ids)).values
            m = (prior_games(c, w) - pwt).clip(lower=0)
            pred = (den * xbar.fillna(0) + m * base) / (den + m)
            t = cur[cur.week == w].set_index("gsis_id").pts
            t = t[t.index.isin(pred.index[den > 0])]
            err += float(((t - pred.reindex(t.index)) ** 2).sum())
            n += len(t)
        return err / n

    grid = itertools.product([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
                             [1, 2, 3, 4, 6, 8, 12, 17], [0, 0.5, 1, 2, 4])
    res = sorted((mse(d, p, f), d, p, f) for d, p, f in grid if f <= p)
    now = mse(cfg["projection"]["recency_decay"], cfg["projection"]["prior_games"],
              cfg["projection"]["prior_games_floor"])
    print(f"one-game-ahead MSE on {season} (prior {season - 1}): current config {now:.2f}")
    for e, d, p, f in res[:5]:
        print(f"  MSE {e:.2f}  recency_decay = {d}  prior_games = {p}  prior_games_floor = {f}")


def fit_injuries() -> None:
    """Empirical P(player appears in the game | final injury-report status)."""
    seasons = list(range(2021, nv.current_season()))
    inj = (nv.nfl.load_injuries(seasons)
           .filter(pl.col("game_type") == "REG", pl.col("position").is_in(nv.POSITIONS),
                   pl.col("report_status").is_in(["Questionable", "Doubtful", "Out"]))
           .select("season", "week", "gsis_id", "report_status").to_pandas())
    ppl = nv.players()
    snaps = pd.concat([nv.snap_counts(s) for s in seasons])
    snaps["gsis_id"] = snaps.pfr_id.map(ppl.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
                                        .set_index("pfr_id").gsis_id)
    stats = pd.concat([nv.player_stats(s) for s in seasons])
    played = pd.concat([snaps[snaps.offense_snaps > 0][["season", "week", "gsis_id"]],
                        stats[["season", "week", "gsis_id"]]]).drop_duplicates()
    played["played"] = True
    inj = inj.merge(played, on=["season", "week", "gsis_id"], how="left").fillna({"played": False})
    out = inj.groupby("report_status").played.agg(["mean", "size"])
    print(f"P(plays | status), {seasons[0]}-{seasons[-1]} regular season, QB/RB/WR/TE:")
    for status, r in out.iterrows():
        print(f"  {status:<13} {r['mean']:.3f}   (n = {int(r['size'])})")


def fit_values(cfg: dict, mode: str, weeks: list[int], seasons: list[int]) -> None:
    if mode == "market":
        raw = [(inp, inp.market.set_index("gsis_id").fantasycalc)
               for inp in (load_inputs(cfg, s, w) for s, w in itertools.product(seasons, weeks))]
    else:
        raw = backtest_cases(cfg, seasons, weeks)
    cases = [(inp, player_pool(inp, cfg), 100 * t / t.max()) for inp, t in raw]

    def values(c: dict) -> list[tuple[pd.Series, pd.Series]]:
        out = []
        for inp, pool, tgt in cases:
            v = run_model(inp, c, pool).value
            both = v.index.intersection(tgt.index)
            out.append((v[both], tgt[both]))
        return out

    def loss(x: np.ndarray) -> float:
        return float(np.mean([((v - t) ** 2).mean() for v, t in values(apply(cfg, x))]))

    x0 = np.array([cfg[s][k] for s, k, _, _ in PARAMS], float)
    res = minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-3})
    best = apply(cfg, res.x)
    rho = np.mean([spearmanr(v, t).statistic for v, t in values(best)])
    print(f"{mode} {seasons} weeks {weeks}: MSE {loss(x0):.1f} -> {res.fun:.1f}   mean Spearman {rho:.3f}\n")
    print("Suggested config values:")
    for sec, key, _, _ in PARAMS + [("layer1", "b2", 0, 0)]:
        print(f"  [{sec}] {key} = {best[sec][key]:.3f}")


def backtest_cases(cfg: dict, seasons: list[int], weeks: list[int]) -> list:
    """(inputs, realized ROS VORP) per season-week, with today's Sleeper data stripped."""
    cases = []
    for season, week in itertools.product(seasons, weeks):
        inp = load_inputs(cfg, season, week)
        inp.meta[["sleeper_id", "sleeper_team", "injury_status", "sleeper_position"]] = np.nan
        inp.market = inp.market.iloc[0:0]
        cases.append((inp, realized_vorp(season, week, cfg).clip(lower=0)))
    return cases


def benchmark(cfg: dict, seasons: list[int], weeks: list[int]) -> None:
    """Spearman vs realized ROS VORP, overall and among the top 60 (either list), vs naive baselines."""
    res: dict[str, list[float]] = {}
    for inp, t in backtest_cases(cfg, seasons, weeks):
        v = run_model(inp, cfg).value
        b = v.index.intersection(t.index)
        top = v[b].nlargest(60).index.union(t[b].nlargest(60).index)
        cur = inp.panel[inp.panel.season == inp.season].groupby("gsis_id").points.mean()
        prev = inp.panel[inp.panel.season < inp.season].groupby("gsis_id").points.mean()
        for name, x in (("model", v), ("PPG so far", cur), ("last-season PPG", prev)):
            for scope, ids in (("all", b), ("top-60", top)):
                rho = spearmanr(x.reindex(ids).fillna(0), t[ids]).statistic
                res.setdefault(f"{name:<16} {scope}", []).append(rho)
    print(f"Spearman vs realized ROS VORP, {seasons} weeks {weeks}:")
    for k, x in res.items():
        print(f"  {k:<24} {np.mean(x):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["forecast", "injuries", "backtest", "market", "benchmark"], default="forecast")
    ap.add_argument("--weeks", type=int, nargs="+", default=[4, 6, 8, 10])
    ap.add_argument("--season", type=int, nargs="+",
                    help="season(s) to fit on, pooled (default: last season; market: current)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cfg = load_config(ROOT / "config.toml")
    current = cfg["league"]["season"] or nv.current_season()
    seasons = args.season or [current if args.mode == "market" else current - 1]
    if args.mode == "forecast":
        for season in seasons:
            fit_forecast(cfg, season)
    elif args.mode == "injuries":
        fit_injuries()
    elif args.mode == "benchmark":
        benchmark(cfg, seasons, args.weeks)
    else:
        fit_values(cfg, args.mode, args.weeks, seasons)


if __name__ == "__main__":
    main()

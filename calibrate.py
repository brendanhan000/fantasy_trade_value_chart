"""Fit b1, lambda, g1, g2, k (optional; prints suggested config values, never edits config).

  python calibrate.py --mode market   --week 6   # fit model-only values to FantasyCalc
  python calibrate.py --mode backtest --week 6   # fit last season's week-6 values to realized ROS VORP

Both modes compare on a 0-100 scale with alpha = 1 (model only), so the market
isn't fitted to itself.
"""
from __future__ import annotations

import argparse
import copy
import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from main import ROOT, load_config
from data import nflverse as nv
from model import scoring as sc
from model.features import load_inputs
from model.layers.l3_vorp import replacement_levels
from model.pipeline import player_pool, run_model

PARAMS = [("layer1", "b1", 0.0, 1.0), ("layer2", "lambda", 0.0, 0.5),
          ("layer2", "g1", 0.0, 6.0), ("layer2", "g2", 0.0, 6.0), ("layer6", "k", 1.0, 2.0)]


def apply(cfg: dict, x: np.ndarray) -> dict:
    c = copy.deepcopy(cfg)
    for (sec, key, lo, hi), v in zip(PARAMS, x):
        c[sec][key] = float(np.clip(v, lo, hi))
    c["layer1"]["b2"] = 1 - c["layer1"]["b1"]
    c["layer6"]["alpha"] = 1.0
    return c


def realized_vorp(season: int, week: int, cfg: dict) -> pd.Series:
    """Actual rest-of-season points over replacement, same replacement rule as the model."""
    hours = cfg["data"]["cache_hours"]
    s = nv.player_stats(season, hours)
    s = s[s.week.between(week, cfg["league"]["last_week"])]
    s = s.merge(nv.pbp_player_week(season, hours), on=["season", "week", "gsis_id"], how="left")
    s["pts"] = (sc.base_points(s, cfg["scoring"]) + sc.milestone_points(s, cfg["scoring"])
                + sc.long_td_points(s, cfg["scoring"]))
    omega = {int(w): v for w, v in cfg["layer3"]["week_weights"].items()}
    parts = []
    for w, g in s.groupby("week"):
        g = g.set_index("gsis_id")
        R = replacement_levels(g.pts, g.position, cfg)
        parts.append((g.pts - g.position.map(R)) * omega.get(int(w), 1.0))
    return pd.concat(parts).groupby(level=0).sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["market", "backtest"], default="market")
    ap.add_argument("--week", type=int, default=6)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cfg = load_config(ROOT / "config.toml")
    season = cfg["league"]["season"] or nv.current_season()
    if args.mode == "backtest":
        season -= 1
    inp = load_inputs(cfg, season, args.week)

    if args.mode == "market":
        tgt = inp.market.set_index("gsis_id").fantasycalc
    else:
        # Today's Sleeper team/injury data would leak into last season; use box-score teams.
        inp.meta[["sleeper_id", "sleeper_team", "injury_status", "sleeper_position"]] = np.nan
        inp.market = inp.market.iloc[0:0]
        tgt = realized_vorp(season, args.week, cfg).clip(lower=0)
    tgt = 100 * tgt / tgt.max()
    pool = player_pool(inp, cfg)

    def loss(x: np.ndarray) -> float:
        v = run_model(inp, apply(cfg, x), pool).value
        both = v.index.intersection(tgt.index)
        return float(((v[both] - tgt[both]) ** 2).mean())

    x0 = np.array([cfg[s][k] for s, k, _, _ in PARAMS], float)
    res = minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-3})
    best = apply(cfg, res.x)
    v = run_model(inp, best, pool).value
    both = v.index.intersection(tgt.index)
    print(f"{args.mode}: MSE {loss(x0):.1f} -> {res.fun:.1f}   "
          f"Spearman {spearmanr(v[both], tgt[both]).statistic:.3f} on {len(both)} players\n")
    print("Suggested config values:")
    for sec, key, _, _ in PARAMS + [("layer1", "b2", 0, 0)]:
        print(f"  [{sec}] {key} = {best[sec][key]:.3f}")


if __name__ == "__main__":
    main()

"""Layer 3: rest-of-season points over replacement.

V_season = sum_w omega_w * a_w * (P_play_w - R_pos_w) * S_pos

a_w multiplies the margin, not just the points: a week a player misses is a
week you start the replacement, which is worth zero over replacement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POSITIONS = ["QB", "RB", "WR", "TE"]


def starters(cfg: dict) -> dict[str, int]:
    return {p: cfg["lineup"][p] * cfg["league"]["teams"] for p in POSITIONS}


def replacement_levels(pts: pd.Series, pos: pd.Series, cfg: dict) -> dict[str, float]:
    """One week: dedicated starters, then FLEX from the rest, then next man up per position."""
    l3, flex_pos = cfg["layer3"], cfg["lineup"]["flex_eligible"]
    order = pts.sort_values(ascending=False)
    pos = pos.reindex(order.index)
    used = starters(cfg)
    taken = pd.Series(False, index=order.index)
    for p, n in used.items():
        taken[pos[pos == p].index[:n]] = True
    for gid in order.index[~taken.values & pos.isin(flex_pos).values][:cfg["lineup"]["FLEX"] * cfg["league"]["teams"]]:
        taken[gid] = True
        used[pos[gid]] += 1

    out = {}
    for p in POSITIONS:
        ranked = order[pos == p]
        starter = ranked.iloc[min(used[p], len(ranked) - 1)]
        waiver = ranked.iloc[min(l3["waiver_rank"][p] - 1, len(ranked) - 1)]
        out[p] = l3["starter_weight"] * starter + (1 - l3["starter_weight"]) * waiver
    return out


def scarcity(mean_pts: pd.Series, pos: pd.Series, cfg: dict) -> dict[str, float]:
    """Points lost per rank across each position's dedicated starters, relative to the average."""
    l3 = cfg["layer3"]
    slope = {}
    for p, n in starters(cfg).items():
        top = mean_pts[pos == p].nlargest(n).values
        slope[p] = -np.polyfit(np.arange(len(top)), top, 1)[0] if len(top) > 1 else 0.0
    mean_slope = np.mean(list(slope.values()))
    return {p: (s / mean_slope) ** l3["scarcity_exp"] if s > 0 else 1.0 for p, s in slope.items()}


def season_value(p_play: pd.DataFrame, avail: pd.DataFrame, pos: pd.Series,
                 cfg: dict) -> tuple[pd.Series, pd.DataFrame, dict[str, float]]:
    """Returns (V_season, replacement level per week x position, S_pos)."""
    omega = {int(w): v for w, v in cfg["layer3"]["week_weights"].items()}
    R = pd.DataFrame({w: replacement_levels(p_play[w], pos, cfg) for w in p_play.columns}).T
    S = scarcity(p_play.mean(axis=1), pos, cfg)
    margin = p_play - R.reindex(columns=pos.values).T.set_axis(p_play.index)
    weights = pd.Series([omega.get(int(w), 1.0) for w in p_play.columns], index=p_play.columns)
    v = (avail * margin).mul(weights, axis=1).sum(axis=1) * pos.map(S)
    return v, R, S

"""Layer 2: risk adjustment.

P_star_w = a_w * P_play_w, where
  a_w      = p_health_w * (1 - bye_w)                       (does he play?)
  P_play_w = P_hat_w * r - lambda*sigma + g1*Boom - g2*Bust  (points if he plays)

The spec writes the additive risk terms outside the availability factor; they
are scaled by a_w here so a player on bye doesn't collect boom points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _shrink(x: pd.Series, n: pd.Series, k: float, pos: pd.Series) -> pd.Series:
    prior = x.groupby(pos).transform("median")
    return (n * x.fillna(prior) + k * prior) / (n + k)


def availability(a: pd.DataFrame, opponents: pd.DataFrame, panel: pd.DataFrame,
                 team_games: dict, season: int, cfg: dict) -> pd.DataFrame:
    l2 = cfg["layer2"]
    played = panel.groupby(["gsis_id", "season"]).size().unstack(fill_value=0)
    played = played.reindex(a.index, fill_value=0)
    prev_games = (played.get(season - 1, 0) > 0) * 17
    cur_games = a.team.map(lambda t: team_games.get((season, t), 0))
    possible = (prev_games + cur_games).clip(lower=1)
    k = l2["availability_prior_games"]
    base = ((played.sum(axis=1) + k * l2["availability_prior"]) / (possible + k)).clip(upper=1.0)

    p = pd.DataFrame(np.repeat(base.values[:, None], len(opponents.columns), axis=1),
                     index=a.index, columns=opponents.columns)
    for gid, status in a.injury_status.dropna().items():
        if status in l2["status"]:
            prob, n_weeks = l2["status"][status]
            p.loc[gid, opponents.columns[:int(n_weeks)]] = prob
    bye = opponents.reindex(a.team).set_index(a.index).isna()
    return p * (~bye)


def role_security(a: pd.DataFrame, cfg: dict) -> pd.Series:
    d = cfg["layer2"]["depth"]
    bonus = d["draft_round_bonus"]
    out = []
    for pos, rank, rnd in zip(a.position, a.depth_rank, a.draft_round):
        ladder = d[pos]
        r = 1.0 if pd.isna(rank) else ladder[min(int(rank), len(ladder)) - 1]
        if pd.notna(rnd) and 1 <= rnd <= len(bonus):
            r += bonus[int(rnd) - 1]
        out.append(min(r, 1.0))
    return pd.Series(out, index=a.index)


def risk_adjust(p_hat: pd.DataFrame, a: pd.DataFrame, avail: pd.DataFrame,
                cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (P_play by week, components)."""
    l2 = cfg["layer2"]
    k = l2["sigma_prior_games"]
    c = pd.DataFrame(index=a.index)
    c["r"] = role_security(a, cfg)
    c["sigma"] = np.sqrt(_shrink(a.sigma**2, a.n_eff, k, a.position))
    c["boom"] = _shrink(a.boom, a.n_eff, k, a.position)
    c["bust"] = _shrink(a.bust, a.n_eff, k, a.position)
    c["risk_pts"] = -l2["lambda"] * c.sigma + l2["g1"] * c.boom - l2["g2"] * c.bust
    c["avail_mean"] = avail.mean(axis=1)
    p_play = p_hat.mul(c.r, axis=0).add(c.risk_pts, axis=0)
    return p_play, c

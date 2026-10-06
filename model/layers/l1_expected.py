"""Layer 1: expected weekly points.

P_hat_w = [b1 * Proj + b2 * (O * F * E)] * SOS_w + MilestoneBonus + LongTDBonus

O*F*E is turned into points by scaling a reference projection: the mean Proj of
the position's top `ref_pool` players. O = 1.0 means typical-starter usage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model import scoring as sc
from model.features import YARD_TYPES


def _z(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0


def _blend(df: pd.DataFrame, weights: dict[str, float], absolute: bool) -> pd.Series:
    """Weighted mean of available (non-NaN) columns; missing ones drop out of the denominator."""
    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        if col not in df:
            continue
        ok = df[col].notna()
        num += df[col].fillna(0) * w
        den += ok * (abs(w) if absolute else w)
    return (num / den.replace(0, np.nan)).fillna(0.0)


def stat_projection(a: pd.DataFrame, scoring: dict) -> pd.Series:
    return sc.base_points(a, scoring)


def milestone_bonus(a: pd.DataFrame, scoring: dict, k: float) -> pd.Series:
    total = pd.Series(0.0, index=a.index)
    for col in YARD_TYPES:
        sd, n = a[f"{col}_logsd"], a[f"{col}_n"].fillna(0)
        prior = sd[n >= 4].groupby(a.position).median().reindex(a.position).values
        prior = np.where(np.isnan(prior), 0.6, prior)
        sigma = np.sqrt((n * sd.fillna(0) ** 2 + k * prior**2) / (n + k))
        total += sc.expected_milestones(a[col].fillna(0).values, sigma, scoring["milestones"][col])
    return total


def long_td_bonus(a: pd.DataFrame, scoring: dict, k: float) -> pd.Series:
    cfg = scoring["long_td"]
    total = pd.Series(0.0, index=a.index)
    for kind, prefix, td_col in (("rushing", "rush_td", "rushing_tds"),
                                 ("receiving", "rec_td", "receiving_tds")):
        tiers = cfg[kind]
        n_td = a[f"{prefix}_n"].fillna(0)
        expl = a[f"{prefix}_explosive"]
        pos_expl = expl.groupby(a.position).transform("median")
        tilt = (expl / pos_expl).clip(0.5, 2.0).fillna(1.0)
        pis = []
        for t, _ in tiers:
            hits = a[f"{prefix}_{int(t)}"].fillna(0)
            base = (hits.groupby(a.position).transform("sum")
                    / n_td.groupby(a.position).transform("sum")).fillna(0)
            pi = (hits + k * base * tilt) / (n_td + k)
            pis.append(pi.clip(0, 1).values)
        for j in range(1, len(pis)):  # P(>= 50) can't exceed P(>= 40)
            pis[j] = np.minimum(pis[j], pis[j - 1])
        total += sc.expected_long_td(a[td_col].fillna(0).values, pis, tiers, cfg["stack"])
    return total


def expected_points(a: pd.DataFrame, team_env: pd.DataFrame, def_ratio: pd.DataFrame,
                    opponents: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (P_hat by week [players x weeks], component table)."""
    l1, scoring = cfg["layer1"], cfg["scoring"]
    c = pd.DataFrame(index=a.index)
    c["proj"] = stat_projection(a, scoring)

    O, F, E, ref = (pd.Series(np.nan, index=a.index) for _ in range(4))
    env = team_env.reindex(a.team).set_index(a.index)
    for pos, idx in a.groupby("position").groups.items():
        g = a.loc[idx]
        pool = c.proj[idx].nlargest(l1["ref_pool"][pos]).index
        ow = l1["opportunity"][pos]
        ratios = pd.DataFrame({k: g[k] / g.loc[pool, k].mean() for k in ow})
        O[idx] = _blend(ratios, ow, absolute=False)
        fz = pd.DataFrame({k: _z(g[k]) for k in l1["efficiency"][pos]})
        F[idx] = 1 + (l1["f_scale"] * _blend(fz, l1["efficiency"][pos], True)).clip(-l1["f_cap"], l1["f_cap"])
        E[idx] = 1 + (l1["e_scale"] * _blend(env.loc[idx], l1["environment"][pos], True)).clip(-l1["e_cap"], l1["e_cap"])
        ref[idx] = c.proj[pool].mean()
    c["O"], c["F"], c["E"] = O, F, E
    c["ofe_pts"] = ref * O * F * E
    c["core"] = l1["b1"] * c.proj + l1["b2"] * c.ofe_pts
    c["milestone"] = milestone_bonus(a, scoring, l1["milestone_sigma_k"])
    c["long_td"] = long_td_bonus(a, scoring, l1["long_td_prior_tds"])

    opp = opponents.reindex(a.team).set_index(a.index)
    sos = pd.DataFrame(index=a.index, columns=opponents.columns, dtype=float)
    for w in opponents.columns:
        ratio = [def_ratio.at[o, p] if isinstance(o, str) and o in def_ratio.index else 1.0
                 for o, p in zip(opp[w], a.position)]
        sos[w] = 1 + l1["sos_strength"] * (np.array(ratio) - 1)
    c["sos_mean"] = sos.mean(axis=1)
    p_hat = sos.mul(c.core, axis=0).add(c.milestone + c.long_td, axis=0)
    return p_hat, c

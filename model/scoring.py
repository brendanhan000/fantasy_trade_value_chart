"""League scoring: base stat line, yardage milestones, long-TD bonuses."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

_NON_STAT = {"milestones", "long_td"}


def base_points(stats: pd.DataFrame, scoring: dict) -> pd.Series:
    """Dot product of stat columns with the scoring vector."""
    total = pd.Series(0.0, index=stats.index)
    for col, pts in scoring.items():
        if col not in _NON_STAT and col in stats:
            total += stats[col].fillna(0).astype(float) * pts
    return total


def milestone_points(stats: pd.DataFrame, scoring: dict) -> pd.Series:
    """Actual per-game yardage bonuses."""
    total = pd.Series(0.0, index=stats.index)
    for col, tiers in scoring["milestones"].items():
        y = stats[col].fillna(0)
        for lo, hi, pts in tiers:
            total += ((y >= lo) & (y < hi)) * pts
    return total


def long_td_award(lengths_reached: list[bool], tiers: list, stack: bool) -> float:
    """Points for one TD given which thresholds its length reached."""
    hit = [pts for reached, (_, pts) in zip(lengths_reached, tiers) if reached]
    if not hit:
        return 0.0
    return float(sum(hit)) if stack else float(max(hit))


def long_td_points(counts: pd.DataFrame, scoring: dict) -> pd.Series:
    """Actual long-TD bonuses from per-game counts (rush_td_40, rush_td_50, rec_td_40, ...).

    Counts are cumulative (a 55-yd TD is in both the 40 and 50 counts), so
    TDs that reached exactly tier j and no higher = n_j - n_{j+1}.
    """
    cfg = scoring["long_td"]
    total = pd.Series(0.0, index=counts.index)
    for kind, prefix in (("rushing", "rush_td"), ("receiving", "rec_td")):
        tiers = cfg[kind]
        n = [counts.get(f"{prefix}_{int(t)}", pd.Series(0, index=counts.index)).fillna(0)
             for t, _ in tiers] + [0]
        for j in range(len(tiers)):
            reached = [i <= j for i in range(len(tiers))]
            total += (n[j] - n[j + 1]) * long_td_award(reached, tiers, cfg["stack"])
    return total


def expected_milestones(mean_yds: np.ndarray, sigma: np.ndarray, tiers: list) -> np.ndarray:
    """E[bonus] with per-game yards ~ lognormal(mean=mean_yds, log-sd=sigma)."""
    mean_yds = np.maximum(np.asarray(mean_yds, float), 1e-6)
    mu = np.log(mean_yds) - sigma**2 / 2
    cdf = lambda y: norm.cdf((np.log(y) - mu) / sigma) if np.isfinite(y) else 1.0  # noqa: E731
    return sum(pts * (cdf(hi) - cdf(lo)) for lo, hi, pts in tiers)


def expected_long_td(exp_tds: np.ndarray, pis: list[np.ndarray], tiers: list,
                     stack: bool) -> np.ndarray:
    """E[TDs] * E[bonus per TD]; pis[j] = P(TD length >= tier j threshold)."""
    pis = pis + [np.zeros_like(pis[0])]
    per_td = sum((pis[j] - pis[j + 1]) *
                 long_td_award([i <= j for i in range(len(tiers))], tiers, stack)
                 for j in range(len(tiers)))
    return exp_tds * per_td

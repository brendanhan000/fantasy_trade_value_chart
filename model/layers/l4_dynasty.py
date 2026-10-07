"""Layer 4: multi-year value. Skipped when horizon T = 0 (redraft).

V_model = V_season + sum_{t=1..T} delta^t * A_pos(age+t) * C_t * D * V_season
"""
from __future__ import annotations

import pandas as pd


def aging(pos: str, age: float, curves: dict) -> float:
    end, decline = curves[pos]
    if pd.isna(age) or age <= end:
        return 1.0
    return max(0.0, 1 - decline * (age - end))


def multi_year(v_season: pd.Series, meta: pd.DataFrame, season: int, cfg: dict) -> pd.Series:
    T = cfg["league"]["horizon_years"]
    if T <= 0:
        return v_season
    l4 = cfg["layer4"]
    rounds = l4["draft_round_mult"]
    extra = []
    for gid, v in v_season.items():
        m = meta.loc[gid]
        rnd = m.get("draft_round")
        D = rounds[int(rnd) - 1] if pd.notna(rnd) and 1 <= rnd <= len(rounds) else 0.95
        end = m.get("contract_end")
        total = 0.0
        for t in range(1, T + 1):
            C = 1.0 if pd.isna(end) or season + t <= end else l4["free_agent_mult"]
            total += l4["delta"] ** t * aging(m.position, m.age + t, l4["aging"]) * C * D
        extra.append(total * v)
    return v_season + pd.Series(extra, index=v_season.index)

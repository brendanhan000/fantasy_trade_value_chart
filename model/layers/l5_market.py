"""Layer 5: market value. Weighted z-scores of each source, rescaled to the model's scale."""
from __future__ import annotations

import pandas as pd


def market_value(v_model: pd.Series, market: pd.DataFrame, cfg: dict) -> pd.Series:
    m = market.set_index("gsis_id").reindex(v_model.index)
    num = pd.Series(0.0, index=v_model.index)
    den = pd.Series(0.0, index=v_model.index)
    for src, w in cfg["layer5"]["sources"].items():
        if src not in m or m[src].notna().sum() < 3:
            continue
        z = (m[src] - m[src].mean()) / m[src].std()
        num += z.fillna(0) * w
        den += z.notna() * w
    mz = num / den.where(den > 0)
    both = mz.notna() & v_model.notna()
    if both.sum() < 3:
        return mz * float("nan")
    # Match mean/sd on the players both sources cover.
    vm, mzb = v_model[both], mz[both]
    return vm.mean() + (mz - mzb.mean()) / mzb.std() * vm.std()

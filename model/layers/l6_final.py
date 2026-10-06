"""Layer 6: V = [alpha * V_model + (1 - alpha) * M] ^ k, rescaled to 0-100.

Below-replacement blends floor at 0 before the power (a negative base has no
real k-th power). A player missing one side uses the other alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def final_value(v_model: pd.Series, m: pd.Series, cfg: dict) -> pd.Series:
    alpha, k = cfg["layer6"]["alpha"], cfg["layer6"]["k"]
    blend = (alpha * v_model + (1 - alpha) * m).fillna(v_model).fillna(m)
    v = np.maximum(blend, 0) ** k
    return 100 * v / v.max() if v.max() > 0 else v

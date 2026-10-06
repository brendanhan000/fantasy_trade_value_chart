"""Scoring + replacement-level checks. Run: python -m pytest tests  (or python tests/test_core.py)"""
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model import scoring as sc  # noqa: E402
from model.layers.l3_vorp import replacement_levels  # noqa: E402

CFG = tomllib.loads((ROOT / "config.toml").read_text())
S = CFG["scoring"]


def test_base_and_milestones():
    g = pd.DataFrame({"passing_yards": [312], "passing_tds": [2], "passing_interceptions": [1],
                      "rushing_yards": [104], "receiving_yards": [0], "receptions": [0]})
    assert np.isclose(sc.base_points(g, S)[0], 312 * .04 + 8 - 2 + 10.4)
    assert sc.milestone_points(g, S)[0] == 2  # 300-399 pass (+1) and 100-199 rush (+1)
    assert sc.milestone_points(g.assign(rushing_yards=200), S)[0] == 3


def test_long_td_stacking():
    tiers = S["long_td"]["rushing"]
    counts = pd.DataFrame({"rush_td_40": [2], "rush_td_50": [1]})  # one 45-yd, one 55-yd TD
    stack = {**S, "long_td": {**S["long_td"], "stack": True}}
    nostack = {**S, "long_td": {**S["long_td"], "stack": False}}
    assert sc.long_td_points(counts, stack)[0] == 1 + 2.5
    assert sc.long_td_points(counts, nostack)[0] == 1 + 1.5
    # Expected: 1 TD, P(>=40)=0.5, P(>=50)=0.2 -> 0.3*1 + 0.2*2.5
    e = sc.expected_long_td(np.array([1.0]), [np.array([.5]), np.array([.2])], tiers, True)
    assert np.isclose(e[0], 0.3 + 0.5)


def test_expected_milestone_lognormal():
    tiers = S["milestones"]["rushing_yards"]
    low, high = sc.expected_milestones(np.array([30.0, 150.0]), np.array([0.6, 0.6]), tiers)
    assert low < 0.05 < 0.5 < high < 2


def test_replacement_levels_with_flex():
    pos = pd.Series(["QB"] * 15 + ["RB"] * 40 + ["WR"] * 45 + ["TE"] * 15)
    pts = pd.Series(np.concatenate([np.linspace(25, 10, 15), np.linspace(20, 2, 40),
                                    np.linspace(19, 3, 45), np.linspace(14, 3, 15)]))
    R = replacement_levels(pts, pos, CFG)
    assert R["QB"] < pts[pos == "QB"].iloc[9]          # below QB10
    assert R["RB"] < pts[pos == "RB"].iloc[19]         # below RB20 (flex pulls deeper)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")

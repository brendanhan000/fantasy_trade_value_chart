"""Run layers 1-6 on loaded inputs. Pure given Inputs, so calibration can call it repeatedly."""
from __future__ import annotations

import pandas as pd

from model.features import Inputs, player_aggregates
from model.layers import l1_expected, l2_risk, l3_vorp, l4_dynasty, l5_market, l6_final


def player_pool(inp: Inputs, cfg: dict) -> pd.DataFrame:
    a = player_aggregates(inp.panel, cfg)
    meta = inp.meta.reindex(a.index)
    a["position"] = meta.sleeper_position.fillna(a.position)
    a["team"] = meta.sleeper_team.where(meta.sleeper_team.notna(), a.stat_team)
    # A player Sleeper knows about but lists with no team is a free agent.
    a.loc[meta.sleeper_id.notna() & meta.sleeper_team.isna(), "team"] = None
    for col in ["injury_status", "depth_rank", "draft_round", "age", "contract_end"]:
        a[col] = meta[col] if col in meta else pd.NA

    drop = [(a.at[g, "name"], "too little game data") for g in a.index[a.n_eff < cfg["projection"]["min_weight"]]]
    drop += [(a.at[g, "name"], "no current NFL team") for g in a.index[a.team.isna()]]
    inp.missing.extend(drop)
    return a[(a.n_eff >= cfg["projection"]["min_weight"]) & a.team.notna()
             & a.position.isin(["QB", "RB", "WR", "TE"])]


def run_model(inp: Inputs, cfg: dict, pool: pd.DataFrame | None = None) -> pd.DataFrame:
    a = player_pool(inp, cfg) if pool is None else pool
    p_hat, c1 = l1_expected.expected_points(a, inp.team_env, inp.def_ratio, inp.opponents, cfg)
    avail = l2_risk.availability(a, inp.opponents, inp.panel, inp.team_games, inp.season, cfg)
    p_play, c2 = l2_risk.risk_adjust(p_hat, a, avail, cfg)
    v_season, _, S = l3_vorp.season_value(p_play, avail, a.position, cfg)
    v_model = l4_dynasty.multi_year(v_season, a, inp.season, cfg)

    out = pd.concat([a[["name", "position", "team", "age", "injury_status", "depth_rank"]], c1, c2], axis=1)
    out["p_hat_mean"] = p_hat.mean(axis=1)
    out["p_star_mean"] = (avail * p_play).mean(axis=1)
    out["scarcity"] = a.position.map(S)
    out["v_season"], out["v_model"] = v_season, v_model

    # Market-only players: in FantasyCalc but not modeled (rookies with no snaps, etc.).
    extra = inp.market[~inp.market.gsis_id.isin(out.index)].set_index("gsis_id")
    if len(extra):
        meta = inp.meta.reindex(extra.index)
        mo = pd.DataFrame({"name": extra.fc_name, "position": meta.sleeper_position,
                           "team": meta.sleeper_team}, index=extra.index)
        mo = mo[mo.position.isin(["QB", "RB", "WR", "TE"]) & mo.team.notna()]
        inp.missing.extend((n, "no model data; market value only") for n in mo.name)
        out = pd.concat([out, mo])

    out["market"] = l5_market.market_value(out.v_model, inp.market, cfg)
    out["value"] = l6_final.final_value(out.v_model, out.market, cfg)
    return out.sort_values("value", ascending=False)

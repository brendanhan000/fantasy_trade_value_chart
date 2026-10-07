"""Turn raw tables into one row per player (and per team / defense).

Every per-game number is a weighted average over games: this season's games
decay by recency, last season's games count for at most `prior_games(cfg, week)` games in
total. That single weighting is both the "recency-weighted average" and the
"regress toward last season" step.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data import apis, cache
from data import nflverse as nv
from model import scoring as sc

log = logging.getLogger(__name__)
POSITIONS = nv.POSITIONS

# Minimum weighted denominators before an efficiency rate is trusted (else neutral).
MIN_SAMPLE = {"dropbacks": 50, "carries": 20, "targets": 10, "routes": 40, "cp_n": 10}
YARD_TYPES = ["passing_yards", "rushing_yards", "receiving_yards"]


def prior_games(cfg: dict, week: int) -> float:
    """Last season's total weight, in games: `prior_games` at week 1, fading linearly
    to `prior_games_floor` by `prior_fade_week`, so 2025 matters less as 2026 accumulates."""
    pj = cfg["projection"]
    fade = max(0.0, 1 - (week - 1) / (pj["prior_fade_week"] - 1))
    return pj["prior_games_floor"] + (pj["prior_games"] - pj["prior_games_floor"]) * fade


@dataclass
class Inputs:
    season: int
    week: int
    weeks: list[int]
    panel: pd.DataFrame            # one row per player-game
    meta: pd.DataFrame             # one row per player: name, pos, team, age, draft, status...
    team_env: pd.DataFrame         # one row per team: environment z-scores
    def_ratio: pd.DataFrame        # index team, columns position: points-allowed ratio
    opponents: pd.DataFrame        # index team, columns week: opponent or NaN (bye)
    team_games: dict[tuple[int, str], int]
    market: pd.DataFrame           # gsis_id + one column per market source
    missing: list[tuple[str, str]] = field(default_factory=list)


def _panel(season: int, week: int, cfg: dict, ppl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = [season - 1, season]
    stats = pd.concat([nv.player_stats(s) for s in seasons], ignore_index=True)
    stats = stats[(stats.season < season) | (stats.week < week)]
    pbp = pd.concat([nv.pbp_player_week(s) for s in seasons], ignore_index=True)
    pfr_map = ppl.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id").gsis_id
    snaps = pd.concat([nv.snap_counts(s) for s in seasons], ignore_index=True)
    rush = pd.concat([nv.pfr_rushing(s) for s in seasons], ignore_index=True)
    for df in (snaps, rush):
        df["gsis_id"] = df.pfr_id.map(pfr_map)
    team = pd.concat([nv.pbp_team_week(s) for s in seasons], ignore_index=True)

    key = ["season", "week", "gsis_id"]
    p = (stats.merge(pbp.drop_duplicates(key), on=key, how="left")
              .merge(snaps.dropna(subset=["gsis_id"]).drop_duplicates(key)[key + ["snap_share", "offense_snaps"]],
                     on=key, how="left")
              .merge(rush.dropna(subset=["gsis_id"]).drop_duplicates(key)[key + ["yards_after_contact"]],
                     on=key, how="left"))
    tw = team.set_index(["season", "week", "team"])
    pass_rate = (tw.dropbacks / tw.plays).rename("team_pass_rate")
    p = p.join(pass_rate, on=["season", "week", "team"])

    p["carry_share"] = p.carries / p.groupby(["season", "week", "team"]).carries.transform("sum")
    p["rz_touches"] = p.rz_targets.fillna(0) + p.rz_carries.fillna(0)
    p["gl_touches"] = p.gl_targets.fillna(0) + p.gl_carries.fillna(0)
    p["routes"] = p.offense_snaps * p.team_pass_rate
    p["cpoe_x_att"] = p.passing_cpoe * p.attempts
    p["att_with_cpoe"] = p.attempts.where(p.passing_cpoe.notna())
    p["carries_with_yac"] = p.carries.where(p.yards_after_contact.notna())
    p["rec_yds_with_routes"] = p.receiving_yards.where(p.routes.notna())
    p["croe_num"] = p.cp_act - p.cp_exp

    scoring = cfg["scoring"]
    p["points"] = (sc.base_points(p, scoring) + sc.milestone_points(p, scoring)
                   + sc.long_td_points(p, scoring))
    p["pos_rank"] = p.groupby(["season", "week", "position"]).points.rank(ascending=False, method="first")
    p["boom"] = (p.pos_rank <= cfg["layer2"]["boom_rank"]).astype(float)
    p["bust"] = (p.pos_rank > cfg["layer2"]["bust_rank"]).astype(float)

    # Game weights.
    pj = cfg["projection"]
    cur = p.season == season
    p["w"] = np.where(cur, pj["recency_decay"] ** (week - 1 - p.week), np.nan)
    n_prev = p[~cur].groupby("gsis_id").week.transform("count")
    p.loc[~cur, "w"] = np.minimum(1.0, prior_games(cfg, week) / n_prev)
    return p, team


def _wsum(p: pd.DataFrame, col: str) -> pd.Series:
    return (p[col] * p.w).groupby(p.gsis_id).sum(min_count=1)


def _wmean(p: pd.DataFrame, col: str) -> pd.Series:
    ok = p[col].notna()
    return _wsum(p[ok], col) / p.w[ok].groupby(p.gsis_id[ok]).sum()


def _wvar(p: pd.DataFrame, x: pd.Series) -> pd.Series:
    """Unbiased weighted variance (Kish effective-n Bessel correction); NaN under 2 effective games."""
    g = p.gsis_id
    sw, sw2 = p.w.groupby(g).sum(), (p.w ** 2).groupby(g).sum()
    mean = (x * p.w).groupby(g).sum() / sw
    raw = ((x - g.map(mean)) ** 2 * p.w).groupby(g).sum() / sw
    n_kish = sw ** 2 / sw2
    return (raw * n_kish / (n_kish - 1)).where(n_kish > 1.5)


PRIOR_FEATURES = ["snap_share", "target_share", "air_yards_share", "carry_share"]


def regression_prior(a: pd.DataFrame, col: str, pos: pd.Series, draft: pd.Series) -> np.ndarray:
    """Prior mean for `col` from usage shares + draft round, fitted by OLS per position on
    established players (8+ games). Usage stabilizes far faster than per-touch efficiency
    or TD rate, so it's the most informative thing a small sample does tell us. Falls back
    to the position median where a position has too few established players."""
    rnd = draft.reindex(a.index)
    X = pd.DataFrame({f: a[f].fillna(0) for f in PRIOR_FEATURES}, index=a.index)
    for r in (1, 2, 3):
        X[f"round_{r}"] = (rnd == r).astype(float)
    X["intercept"] = 1.0
    out = a[col][a.games >= 4].groupby(pos).median().reindex(pos.loc[a.index]).values.copy()
    for p_ in pos.unique():
        is_pos = (pos.reindex(a.index) == p_).values
        train = is_pos & (a.games >= 8).values & a[col].notna().values
        if train.sum() < 3 * X.shape[1]:
            continue
        beta, *_ = np.linalg.lstsq(X.values[train], a[col].values[train], rcond=None)
        out[is_pos] = np.maximum(X.values[is_pos] @ beta, 0)
    return out


def player_aggregates(p: pd.DataFrame, cfg: dict, week: int, draft: pd.Series) -> pd.DataFrame:
    stat_cols = [c for c in cfg["scoring"] if c in p.columns]
    usage = ["snap_share", "target_share", "air_yards_share", "carry_share", "rz_touches",
             "gl_touches", "points", "boom", "bust", "attempts", "carries", "targets"]
    a = pd.DataFrame({c: _wmean(p, c) for c in stat_cols + usage})
    a["n_eff"] = p.w.groupby(p.gsis_id).sum()
    a["games"] = p.groupby("gsis_id").size()

    # Empirical-Bayes shrinkage: every player gets at least `shrink_games` pseudo-games
    # at what their usage and draft capital predict (more if last season is thin), so
    # 2 hot games from a backup don't project as a star and TD luck washes out.
    # Usage shares themselves shrink to the position median.
    pos = p.groupby("gsis_id").position.last()
    prev_w = p.w[p.season < p.season.max()].groupby(p.gsis_id).sum().reindex(a.index, fill_value=0)
    m = (prior_games(cfg, week) - prev_w).clip(lower=cfg["projection"]["shrink_games"])
    pos = pos.loc[a.index]
    priors = {col: regression_prior(a, col, pos, draft) for col in stat_cols}
    for col in usage[:6]:
        priors[col] = a[col][a.games >= 4].groupby(pos).median().reindex(pos).values
    for col, base in priors.items():
        a[col] = (a.n_eff * a[col].fillna(0) + m * base) / (a.n_eff + m)

    def rate(num: str, den: str, min_key: str) -> pd.Series:
        d = _wsum(p, den)
        return (_wsum(p, num) / d).where(d >= MIN_SAMPLE[min_key])
    a["epa_per_dropback"] = rate("dropback_epa", "dropbacks", "dropbacks")
    a["cpoe"] = rate("cpoe_x_att", "att_with_cpoe", "dropbacks")
    a["epa_per_carry"] = rate("rushing_epa", "carries", "carries")
    a["yac_per_carry"] = rate("yards_after_contact", "carries_with_yac", "carries")
    a["epa_per_target"] = rate("receiving_epa", "targets", "targets")
    a["yprr"] = rate("rec_yds_with_routes", "routes", "routes")
    a["croe"] = rate("croe_num", "cp_n", "cp_n")

    # Weekly points spread.
    a["sigma"] = np.sqrt(_wvar(p, p.points))

    # Per-game yardage log-sd for the milestone lognormal (games with >= 1 yard).
    for col in YARD_TYPES:
        q = p[p[col] >= 1]
        a[f"{col}_logsd"] = np.sqrt(_wvar(q, np.log(q[col])))
        a[f"{col}_n"] = q.groupby("gsis_id").size()

    # Long-TD evidence.
    for prefix, td, expl, vol in (("rush_td", "rushing_tds", "rushing_20", "carries"),
                                  ("rec_td", "receiving_tds", "receiving_20", "receptions")):
        a[f"{prefix}_n"] = _wsum(p, td)
        for t, _ in cfg["scoring"]["long_td"]["rushing" if prefix == "rush_td" else "receiving"]:
            a[f"{prefix}_{int(t)}"] = _wsum(p, f"{prefix}_{int(t)}")
        a[f"{prefix}_vol"] = _wsum(p, vol)
        a[f"{prefix}_explosive"] = _wsum(p, expl) / a[f"{prefix}_vol"]

    last = p.sort_values(["season", "week"]).groupby("gsis_id").last()
    a["name"], a["stat_team"], a["position"] = last.name, last.team, last.position
    return a


def team_environment(team: pd.DataFrame, sched: pd.DataFrame, season: int, week: int,
                     cfg: dict, implied: dict[str, float]) -> pd.DataFrame:
    pj = cfg["projection"]
    coach = {}
    for s in (season - 1, season):
        sc_ = sched[sched.season == s]
        coach[s] = dict(zip(sc_.home_team, sc_.home_coach)) | dict(zip(sc_.away_team, sc_.away_coach))
    changed = {t: coach[season].get(t) != coach[season - 1].get(t) for t in coach[season]}

    t = team[(team.season < season) | (team.week < week)].copy()
    cur = t.season == season
    prior_w = min(1.0, prior_games(cfg, week) / 17)
    t["w"] = np.where(cur, pj["recency_decay"] ** (week - 1 - t.week), prior_w)
    t.loc[~cur, "w"] *= t.loc[~cur, "team"].map(
        lambda x: cfg["layer1"]["coach_change_prior_mult"] if changed.get(x) else 1.0)

    g = t.assign(**{c: t[c] * t.w for c in ["plays", "proe", "dropbacks", "sacks", "rushes",
                                            "stuffs", "qb_epa"]}).groupby("team")
    s = g[["plays", "proe", "dropbacks", "sacks", "rushes", "stuffs", "qb_epa", "w"]].sum()
    env = pd.DataFrame({
        "pace": s.plays / s.w,
        "proe": s.proe / s.w,
        "qb_quality": s.qb_epa / s.dropbacks,
        "sack_rate": s.sacks / s.dropbacks,
        "stuff_rate": s.stuffs / s.rushes,
    })
    z = lambda x: (x - x.mean()) / x.std()  # noqa: E731
    out = pd.DataFrame({"pace": z(env.pace), "proe": z(env.proe), "qb_quality": z(env.qb_quality),
                        "ol_quality": z(-(z(env.sack_rate) + z(env.stuff_rate)) / 2)})

    # Vegas: The Odds API if keyed, else nflverse schedule lines for upcoming games, else neutral.
    if not implied:
        up = sched[(sched.season == season) & (sched.week >= week)].dropna(subset=["total_line", "spread_line"])
        home = (up.total_line + up.spread_line) / 2
        away = (up.total_line - up.spread_line) / 2
        implied = pd.concat([pd.Series(home.values, up.home_team.values),
                             pd.Series(away.values, up.away_team.values)]).groupby(level=0).mean().to_dict()
    it = pd.Series(implied, dtype=float).reindex(out.index)
    out["implied_total"] = z(it).fillna(0.0) if it.notna().sum() > 2 else 0.0
    return out.fillna(0.0)


def defense_ratio(p: pd.DataFrame, season: int, week: int, cfg: dict) -> pd.DataFrame:
    """Opponent fantasy points allowed per position, as a ratio to league average, shrunk to 1."""
    pj, k = cfg["projection"], cfg["layer1"]["sos_prior_games"]
    g = p.groupby(["season", "week", "opponent_team", "position"]).points.sum().reset_index()
    cur = g.season == season
    g["w"] = np.where(cur, pj["recency_decay"] ** (week - 1 - g.week), min(1.0, prior_games(cfg, week) / 17))
    g["wp"] = g.points * g.w
    s = g.groupby(["opponent_team", "position"])[["wp", "w"]].sum()
    allowed = s.wp / s.w
    raw = allowed / allowed.groupby(level="position").transform("mean")
    shrunk = (s.w * raw + k) / (s.w + k)
    return shrunk.unstack("position").reindex(columns=POSITIONS).fillna(1.0)


def load_inputs(cfg: dict, season: int, week: int) -> Inputs:
    cache.LIVE_HOURS = cfg["data"]["cache_hours"]
    ppl = nv.players()
    panel, team = _panel(season, week, cfg, ppl)
    sched = nv.schedules([season - 1, season])
    weeks = list(range(week, cfg["league"]["last_week"] + 1))

    implied = apis.odds_implied_totals(cfg["data"]["odds_api_env"], nv.team_names())
    team_env = team_environment(team, sched, season, week, cfg, implied)
    def_ratio = defense_ratio(panel, season, week, cfg)

    cur = sched[sched.season == season]
    opp = pd.concat([cur.rename(columns={"home_team": "team", "away_team": "opp"}),
                     cur.rename(columns={"away_team": "team", "home_team": "opp"})])
    opponents = opp.pivot_table(index="team", columns="week", values="opp", aggfunc="first")
    opponents = opponents.reindex(columns=weeks)
    played = cur[cur.week < week]
    team_games = {(season, t): int(((played.home_team == t) | (played.away_team == t)).sum())
                  for t in opponents.index}

    sl = apis.sleeper_players()
    sl["gsis_id"] = sl.sleeper_id.map(nv.sleeper_to_gsis()).fillna(sl.gsis_id)
    fc = apis.fantasycalc_values(cfg["fantasycalc"])
    market = fc.merge(sl[["sleeper_id", "gsis_id"]], on="sleeper_id", how="left")
    missing = [(r.fc_name, "FantasyCalc player has no gsis_id mapping; dropped")
               for r in market[market.gsis_id.isna()].itertuples()]
    market = market.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")

    meta = (ppl.drop_duplicates("gsis_id").set_index("gsis_id")
               .join(sl.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")))
    meta["age"] = ((pd.Timestamp(f"{season}-09-01") - pd.to_datetime(meta.birth_date, errors="coerce"))
                   .dt.days / 365.25).fillna(meta.sleeper_age)
    meta = meta.join(nv.depth_ranks(season).set_index("gsis_id"))
    if cfg["league"]["horizon_years"] > 0:
        meta = meta.join(nv.contract_end_year().set_index("gsis_id"))

    return Inputs(season, week, weeks, panel, meta, team_env, def_ratio, opponents,
                  team_games, market, missing)

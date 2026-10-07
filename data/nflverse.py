"""nflverse pulls (via nflreadpy, the maintained successor to nfl_data_py).

Everything comes back as pandas, regular season only, keyed by gsis_id where a
player is involved. Play-by-play is heavy, so it is aggregated once per season
into two small tables and cached.
"""
from __future__ import annotations

import logging

import pandas as pd
import polars as pl

from data import cache  # noqa: F401  (configures nflreadpy before import)
import nflreadpy as nfl

log = logging.getLogger(__name__)
POSITIONS = ["QB", "RB", "WR", "TE"]


def current_season() -> int:
    return nfl.get_current_season()


def _hours(season: int) -> float | None:
    return None if season < current_season() else cache.LIVE_HOURS


def _reg(df: pl.DataFrame) -> pl.DataFrame:
    for col in ("season_type", "game_type"):
        if col in df.columns:
            return df.filter(pl.col(col) == "REG")
    return df


def _safe(loader, *args, **kw) -> pl.DataFrame:
    """nflverse raises for seasons it hasn't published; treat that as empty."""
    try:
        return loader(*args, **kw)
    except Exception as exc:  # noqa: BLE001 - nflreadpy raises bare ValueError
        log.warning("%s%s unavailable: %s", loader.__name__, args, str(exc).split("\n")[0])
        return pl.DataFrame()


STAT_COLS = [
    "player_id", "player_display_name", "position", "season", "week", "team", "opponent_team",
    "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "passing_2pt_conversions", "passing_epa", "passing_cpoe", "sacks_suffered",
    "carries", "rushing_yards", "rushing_tds", "rushing_2pt_conversions", "rushing_epa",
    "rushing_20", "targets", "receptions", "receiving_yards", "receiving_tds",
    "receiving_2pt_conversions", "receiving_epa", "receiving_20", "receiving_air_yards",
    "target_share", "air_yards_share",
    "rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost",
]


def player_stats(season: int) -> pd.DataFrame:
    def build() -> pd.DataFrame:
        df = _reg(_safe(nfl.load_player_stats, [season]))
        if df.is_empty():
            return pd.DataFrame(columns=STAT_COLS + ["gsis_id", "name", "fumbles_lost"])
        df = df.filter(pl.col("position").is_in(POSITIONS)).select(
            [c for c in STAT_COLS if c in df.columns])
        out = df.to_pandas().rename(columns={"player_id": "gsis_id",
                                             "player_display_name": "name"})
        fl = [c for c in ("rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost")
              if c in out]
        out["fumbles_lost"] = out[fl].fillna(0).sum(axis=1)
        return out
    return cache.frame(f"stats_{season}", build, _hours(season))


def _pbp(season: int) -> pl.DataFrame:
    p = _reg(_safe(nfl.load_pbp, [season]))
    if p.is_empty():
        return p
    return p.filter(pl.col("play_type").is_in(["pass", "run"]))


def pbp_player_week(season: int) -> pd.DataFrame:
    """Red-zone/goal-line usage, completion-over-expected, long TDs, dropbacks."""
    def build() -> pd.DataFrame:
        p = _pbp(season)
        if p.is_empty():
            return pd.DataFrame(columns=["season", "week", "gsis_id"])
        rz, gl = pl.col("yardline_100") <= 20, pl.col("yardline_100") <= 5
        yds = pl.col("yards_gained")
        has_cp = pl.col("cp").is_not_null()
        rec = (p.filter(pl.col("receiver_player_id").is_not_null())
                .group_by("week", gsis_id="receiver_player_id")
                .agg(rz_targets=rz.sum(), gl_targets=gl.sum(),
                     cp_n=has_cp.sum(),
                     cp_exp=pl.col("cp").filter(has_cp).sum(),
                     cp_act=pl.col("complete_pass").filter(has_cp).sum(),
                     rec_td_40=((pl.col("pass_touchdown") == 1) & (yds >= 40)).sum(),
                     rec_td_50=((pl.col("pass_touchdown") == 1) & (yds >= 50)).sum()))
        rush = (p.filter(pl.col("rusher_player_id").is_not_null())
                 .group_by("week", gsis_id="rusher_player_id")
                 .agg(rz_carries=rz.sum(), gl_carries=gl.sum(),
                      rush_td_40=((pl.col("rush_touchdown") == 1) & (yds >= 40)).sum(),
                      rush_td_50=((pl.col("rush_touchdown") == 1) & (yds >= 50)).sum()))
        drop = (p.filter(pl.col("passer_player_id").is_not_null() & (pl.col("qb_dropback") == 1))
                 .group_by("week", gsis_id="passer_player_id")
                 .agg(dropbacks=pl.len(), dropback_epa=pl.col("qb_epa").sum()))
        out = rec.join(rush, on=["week", "gsis_id"], how="full", coalesce=True)
        out = out.join(drop, on=["week", "gsis_id"], how="full", coalesce=True)
        out = out.with_columns(pl.lit(season).alias("season")).to_pandas()
        return out.fillna(0)
    return cache.frame(f"pbp_player_{season}", build, _hours(season))


def pbp_team_week(season: int) -> pd.DataFrame:
    """Team environment: neutral PROE, pace, QB EPA/dropback, sack and stuff rates."""
    def build() -> pd.DataFrame:
        p = _pbp(season)
        if p.is_empty():
            return pd.DataFrame(columns=["season", "week", "team"])
        p = p.filter(pl.col("posteam").is_not_null())
        neutral = (pl.col("down").is_in([1, 2]) & pl.col("wp").is_between(0.2, 0.8)
                   & (pl.col("half_seconds_remaining") > 120))
        designed_run = (pl.col("play_type") == "run") & (pl.col("qb_scramble") == 0)
        db = pl.col("qb_dropback") == 1
        out = (p.group_by("week", team="posteam")
                .agg(plays=pl.len(),
                     proe=pl.col("pass_oe").filter(neutral).mean() / 100,
                     dropbacks=db.sum(),
                     sacks=pl.col("sack").sum(),
                     rushes=designed_run.sum(),
                     stuffs=(designed_run & (pl.col("yards_gained") <= 0)).sum(),
                     qb_epa=pl.col("qb_epa").filter(db).sum()))
        return out.with_columns(pl.lit(season).alias("season")).to_pandas()
    return cache.frame(f"pbp_team_{season}", build, _hours(season))


def players() -> pd.DataFrame:
    def build() -> pd.DataFrame:
        cols = ["gsis_id", "display_name", "position", "birth_date", "pfr_id",
                "draft_round"]
        return nfl.load_players().select(cols).to_pandas()
    return cache.frame("players", build, 24 * 7)


def snap_counts(season: int) -> pd.DataFrame:
    def build() -> pd.DataFrame:
        df = _reg(_safe(nfl.load_snap_counts, [season]))
        if df.is_empty():
            return pd.DataFrame(columns=["season", "week", "pfr_id", "snap_share", "offense_snaps"])
        return (df.select("season", "week", pl.col("pfr_player_id").alias("pfr_id"),
                          "offense_snaps", pl.col("offense_pct").alias("snap_share"))
                  .to_pandas())
    return cache.frame(f"snaps_{season}", build, _hours(season))


def pfr_rushing(season: int) -> pd.DataFrame:
    def build() -> pd.DataFrame:
        df = _reg(_safe(nfl.load_pfr_advstats, [season], stat_type="rush", summary_level="week"))
        if df.is_empty():
            return pd.DataFrame(columns=["season", "week", "pfr_id", "yards_after_contact"])
        return (df.select("season", "week", pl.col("pfr_player_id").alias("pfr_id"),
                          pl.col("rushing_yards_after_contact").alias("yards_after_contact"))
                  .to_pandas())
    return cache.frame(f"pfr_rush_{season}", build, _hours(season))


def schedules(seasons: list[int]) -> pd.DataFrame:
    def build() -> pd.DataFrame:
        cols = ["season", "week", "home_team", "away_team", "spread_line", "total_line",
                "home_coach", "away_coach"]
        return _reg(nfl.load_schedules(seasons)).select(cols).to_pandas()
    return cache.frame(f"schedules_{'_'.join(map(str, seasons))}", build, cache.LIVE_HOURS)


def depth_ranks(season: int) -> pd.DataFrame:
    """Latest depth-chart rank per player (1 = first string)."""
    def build() -> pd.DataFrame:
        df = _safe(nfl.load_depth_charts, [season])
        if df.is_empty():
            return pd.DataFrame(columns=["gsis_id", "depth_rank"])
        if "pos_abb" in df.columns:  # 2025+: daily ESPN snapshots
            df = df.filter(pl.col("pos_abb").is_in(POSITIONS) & pl.col("gsis_id").is_not_null())
            df = df.filter(pl.col("dt") == pl.col("dt").max().over("team"))
        else:                        # <= 2024: weekly charts with depth_team "1", "2", ...
            df = _reg(df).filter(pl.col("position").is_in(POSITIONS) & pl.col("gsis_id").is_not_null())
            df = df.filter(pl.col("week") == pl.col("week").max().over("club_code")).with_columns(
                pos_rank=pl.col("depth_team").cast(pl.Int32, strict=False))
        return (df.group_by("gsis_id").agg(depth_rank=pl.col("pos_rank").min()).to_pandas())
    return cache.frame(f"depth_{season}", build, cache.LIVE_HOURS)


def contract_end_year() -> pd.DataFrame:
    """Last season covered by each player's active contract (layer 4 only)."""
    def build() -> pd.DataFrame:
        df = nfl.load_contracts().filter(pl.col("is_active") & pl.col("gsis_id").is_not_null())
        return (df.group_by("gsis_id")
                  .agg(contract_end=(pl.col("year_signed") + pl.col("years") - 1).max())
                  .to_pandas())
    return cache.frame("contracts", build, 24 * 7)


def team_names() -> dict[str, str]:
    """Full team name -> nflverse abbreviation (for The Odds API)."""
    t = nfl.load_teams().select("team_abbr", "team_name").to_pandas()
    return dict(zip(t.team_name, t.team_abbr))


def sleeper_to_gsis() -> dict[str, str]:
    """Sleeper id -> gsis_id. Sleeper's own gsis_id field is sparse; ffverse's crosswalk isn't."""
    def build() -> pd.DataFrame:
        return (nfl.load_ff_playerids().select("sleeper_id", "gsis_id")
                   .drop_nulls().to_pandas().astype(str))
    ids = cache.frame("ff_playerids", build, 24 * 7)
    return dict(zip(ids.sleeper_id.str.replace(r"\.0$", "", regex=True), ids.gsis_id))

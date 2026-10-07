"""Sleeper (ids, team, injury status, age), FantasyCalc (market), The Odds API (optional)."""
from __future__ import annotations

import logging
import os

import pandas as pd
import requests

from data import cache

log = logging.getLogger(__name__)
TEAM_FIX = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS"}  # -> nflverse abbreviations


def sleeper_players() -> pd.DataFrame:
    def fetch() -> dict:
        r = requests.get("https://api.sleeper.app/v1/players/nfl", timeout=60)
        r.raise_for_status()
        return r.json()
    raw = cache.blob("sleeper_players", fetch, cache.LIVE_HOURS)
    rows = [{"sleeper_id": str(p.get("player_id")),
             "gsis_id": (p.get("gsis_id") or "").strip() or None,
             "sleeper_name": p.get("full_name"),
             "sleeper_team": TEAM_FIX.get(p.get("team"), p.get("team")),
             "sleeper_age": p.get("age"),
             "injury_status": p.get("injury_status"),
             "sleeper_position": p.get("position")}
            for p in raw.values() if p.get("position") in ("QB", "RB", "WR", "TE")]
    return pd.DataFrame(rows)


def fantasycalc_values(params: dict) -> pd.DataFrame:
    q = {k: str(v).lower() if isinstance(v, bool) else v for k, v in params.items()}

    def fetch() -> list:
        r = requests.get("https://api.fantasycalc.com/values/current", params=q, timeout=30)
        r.raise_for_status()
        return r.json()
    raw = cache.blob("fantasycalc_" + "_".join(f"{k}{v}" for k, v in q.items()), fetch, cache.LIVE_HOURS)
    return pd.DataFrame([{"sleeper_id": str(x["player"].get("sleeperId")),
                          "fc_name": x["player"]["name"],
                          "fantasycalc": float(x["value"])} for x in raw])


def odds_implied_totals(env_var: str, names: dict[str, str]) -> dict[str, float]:
    """Mean implied team total over upcoming games; {} when no key is set."""
    key = os.environ.get(env_var)
    if not key:
        return {}

    def fetch() -> list:
        r = requests.get("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds",
                         params={"apiKey": key, "regions": "us", "markets": "spreads,totals"},
                         timeout=30)
        r.raise_for_status()
        return r.json()
    try:
        games = cache.blob("odds_api", fetch, cache.LIVE_HOURS)
    except requests.RequestException as exc:
        log.warning("Odds API failed, environment uses neutral Vegas term: %s", exc)
        return {}
    totals: dict[str, list[float]] = {}
    for g in games:
        book = next(iter(g.get("bookmakers", [])), None)
        if not book:
            continue
        mk = {m["key"]: m["outcomes"] for m in book["markets"]}
        if "totals" not in mk or "spreads" not in mk:
            continue
        total = mk["totals"][0]["point"]
        for o in mk["spreads"]:
            team = names.get(o["name"])
            if team:  # implied = total/2 - spread/2 (spread is negative for favorites)
                totals.setdefault(team, []).append(total / 2 - o["point"] / 2)
    return {t: sum(v) / len(v) for t, v in totals.items()}

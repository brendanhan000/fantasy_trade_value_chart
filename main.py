"""Fantasy trade value chart.  Usage: python main.py --week 6"""
from __future__ import annotations

import argparse
import logging
import tomllib
from pathlib import Path

from data import cache

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("tvc")


def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--week", type=int, help="first week to value (default: next week to be played)")
    ap.add_argument("--season", type=int, help="default: config league.season, else current")
    ap.add_argument("--config", type=Path, default=ROOT / "config.toml")
    ap.add_argument("--horizon", type=int, help="override league.horizon_years (T)")
    ap.add_argument("--refresh", action="store_true", help="ignore cached downloads")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cache.REFRESH = args.refresh
    from data import nflverse as nv
    from model.features import load_inputs
    from model.pipeline import run_model
    from output.chart import export

    cfg = load_config(args.config)
    if args.horizon is not None:
        cfg["league"]["horizon_years"] = args.horizon
    season = args.season or cfg["league"]["season"] or nv.current_season()
    week = args.week or nv.nfl.get_current_week() + 1
    log.info("valuing %d weeks %d-%d", season, week, cfg["league"]["last_week"])

    inp = load_inputs(cfg, season, week)
    values = run_model(inp, cfg)

    out_dir = ROOT / cfg["chart"]["out_dir"]
    title = f"Trade Value Chart — {season} Week {week}"
    for p in export(values, out_dir, cfg["chart"]["rows"], title):
        log.info("wrote %s", p.relative_to(ROOT))

    miss = out_dir / "missing_data.log"
    miss.write_text("".join(f"{n}\t{why}\n" for n, why in inp.missing))
    if inp.missing:
        log.warning("%d players had missing data (see %s)", len(inp.missing), miss.relative_to(ROOT))


if __name__ == "__main__":
    main()

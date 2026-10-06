# fantasy_trade_value_chart

Rest-of-season trade values for a 10-team ESPN full-PPR league with ESPN's
yardage and long-TD bonuses. Output is a four-column chart (QB / RB / WR / TE),
one row per rank, each cell `Player (TEAM) — value`, scaled 0–100.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # or reuse ../.venv
.venv/bin/python main.py --week 6          # value weeks 6..17 using games through week 5
.venv/bin/python main.py --week 6 --horizon 2   # keeper/dynasty: add 2 future seasons
.venv/bin/python main.py --refresh         # ignore cached downloads
python -m pytest tests                     # scoring + replacement-level checks
```

Writes to `charts/`:

| file | what |
|---|---|
| `trade_value_chart.csv` / `.html` / `.png` | the chart |
| `values_detail.csv` | every player with every layer's components, the file to read when tuning |
| `missing_data.log` | players skipped or market-only, with the reason |

First run downloads about 2 seasons of nflverse data (play-by-play is the large
part). Everything is cached in `.cache/`. Past seasons never expire, and
live-season data expires after `data.cache_hours`.

## Data

| source | used for |
|---|---|
| nflverse via `nflreadpy` (the maintained successor to `nfl_data_py`) | weekly stats, pbp (RZ/GL touches, long TDs, CROE, PROE, pace, OL), snaps, PFR yards after contact, schedules (byes, opponents, lines, coaches), depth charts, draft info, contracts |
| Sleeper API | current team, injury status, age |
| FantasyCalc (`numQbs=1, numTeams=10, ppr=1, isDynasty=false`) | market values |
| The Odds API (only if `ODDS_API_KEY` is set) | Vegas implied team totals |

Without an Odds API key, implied totals come from the nflverse schedule lines
for upcoming games. If those are missing too, the Vegas term is neutral (z = 0).

## Model

Each layer is its own module in `model/layers/`. `model/features.py` builds
the per-player inputs.

1. **Expected points** (`l1_expected.py`): `P_hat_w = [b1·Proj + b2·(O·F·E)]·SOS_w + Milestone + LongTD`.
   - Proj is a recency-weighted per-game stat line times the scoring vector. Every per-game number is a weighted average: this season's games decay by `recency_decay`, and last season counts for at most `prior_games` games. Players with little last-season data get the gap filled with pseudo-games at the position median, so a backup's two hot games don't project as a star.
   - O·F·E is converted to points by scaling the mean Proj of the position's top `ref_pool` players. O = 1 means typical-starter usage. F and E are multipliers built from within-position (or team) z-scores, capped by `f_cap` / `e_cap`.
   - YPRR uses `snaps × team dropback rate` as routes, because free data has no route counts.
   - Milestones: per-game yards are lognormal, with the mean set to the projection and log-sd shrunk toward the position's.
   - Long TDs: π₄₀ / π₅₀ are the player's long-TD share, shrunk toward the position share and tilted by their 20+ yd play rate.
2. **Risk** (`l2_risk.py`): `P_star_w = a_w · (P_hat_w·r − λσ + g1·Boom − g2·Bust)`, where `a_w = p_health_w·(1 − bye_w)`.
   - p_health is the Sleeper status for the next few weeks (`[layer2.status]`), then the player's historical games-played rate.
3. **Points over replacement** (`l3_vorp.py`): replacement is recomputed every week. Dedicated starters are QB10 / RB20 / WR20 / TE10, then FLEX takes the next 10 RB/WR/TE. The level is blended with waiver level (QB15 / RB35 / WR40 / TE14). S_pos comes from the per-rank points drop-off across each position's starters.
4. **Multi-year** (`l4_dynasty.py`): skipped when `horizon_years = 0`.
5. **Market** (`l5_market.py`): z-scored FantasyCalc, mapped onto the model's mean and std.
6. **Final** (`l6_final.py`): `max(α·V_model + (1−α)·M, 0)^k`, rescaled to 0–100. Players below replacement show as 0 and drop off the chart.

Deliberate deviations from the original formula:
- The additive risk terms and the replacement margin are scaled by availability `a_w`. A week a player misses is a week you start the replacement, which is worth 0 over replacement, not −R.
- Replacement ranks use the risk-adjusted points (`P_play`), so the comparison is like-for-like.
- A new head coach halves the weight of last season's team data instead of being its own E term.

## Tuning

Everything is in `config.toml`. The knobs that move the chart most:

| knob | effect |
|---|---|
| `layer6.alpha` | trust in the model vs. FantasyCalc |
| `layer6.k` | consolidation premium. Higher means a bigger gap between stars and depth |
| `layer3.starter_weight`, `layer3.waiver_rank` | how deep "replacement" is. Lower means more players above 0 |
| `layer3.week_weights` | playoff-week emphasis |
| `layer3.scarcity_exp` | extra positional scarcity on top of VORP (0 = off) |
| `projection.prior_games`, `recency_decay` | how fast the model believes new usage |
| `scoring.fumbles_lost`, `scoring.long_td.stack` | check these against your actual ESPN settings |

To see why a player landed where they did, sort `values_detail.csv`. It has
Proj, O/F/E, the bonus terms, availability, r, σ, boom/bust, V_season, and market.

### Calibration (optional)

```bash
python calibrate.py --mode market   --week 6   # fit to FantasyCalc
python calibrate.py --mode backtest --week 6   # fit last season's week-6 values to realized ROS VORP
```

Calibration fits b1 (b2 = 1 − b1), λ, g1, g2 and k with Nelder-Mead and prints
suggested values. It never edits the config. Backtest mode is one season at
one week, so it's noisy. Read its output as a direction, not an answer.

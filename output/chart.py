"""Export the four-column chart as CSV, styled HTML and PNG."""
from __future__ import annotations

import html
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
POSITIONS = ["QB", "RB", "WR", "TE"]


def build_chart(values: pd.DataFrame, rows: int) -> pd.DataFrame:
    cols = {}
    for pos in POSITIONS:
        v = values[(values.position == pos) & (values.value > 0)].head(rows)
        cols[pos] = [f"{n} ({t}) — {x:.1f}" for n, t, x in zip(v.name, v.team, v.value)]
    chart = pd.DataFrame({p: pd.Series(c, dtype=object) for p, c in cols.items()})
    chart.index = pd.RangeIndex(1, len(chart) + 1, name="Rank")
    return chart.fillna("")


def _html(chart: pd.DataFrame, values: pd.DataFrame, title: str) -> str:
    def cell(text: str) -> str:
        if not text:
            return "<td></td>"
        name, val = text.rsplit(" — ", 1)
        alpha = float(val) / 100
        return (f'<td style="background:rgba(37,99,235,{alpha * 0.55:.2f})">'
                f'<span class="n">{html.escape(name)}</span><span class="v">{val}</span></td>')
    body = "\n".join("<tr><th>{}</th>{}</tr>".format(i, "".join(cell(x) for x in r))
                     for i, r in zip(chart.index, chart.values))
    head = "".join(f"<th>{p}</th>" for p in chart.columns)
    return f"""<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font:14px system-ui,sans-serif;margin:24px;background:#fafafa;color:#111}}
table{{border-collapse:collapse;width:100%;max-width:1200px}}
th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left}}
thead th{{background:#111;color:#fff;position:sticky;top:0}}
tbody th{{background:#f0f0f0;width:3em;text-align:right}}
td{{white-space:nowrap}} .v{{float:right;font-weight:600;margin-left:12px}}
</style>
<h1>{html.escape(title)}</h1>
<table><thead><tr><th>#</th>{head}</tr></thead><tbody>
{body}
</tbody></table>"""


def _png(chart: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 0.28 * len(chart) + 1.2))
    ax.axis("off")
    t = ax.table(cellText=chart.values, colLabels=list(chart.columns),
                 rowLabels=[str(i) for i in chart.index], loc="upper center", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.15)
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export(values: pd.DataFrame, out_dir: Path, rows: int, title: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    chart = build_chart(values, rows)
    paths = [out_dir / "trade_value_chart.csv", out_dir / "trade_value_chart.html",
             out_dir / "values_detail.csv"]
    chart.to_csv(paths[0])
    paths[1].write_text(_html(chart, values, title))
    values.round(3).to_csv(paths[2], index_label="gsis_id")
    try:
        png = out_dir / "trade_value_chart.png"
        _png(chart, png, title)
        paths.append(png)
    except ImportError:
        log.warning("matplotlib not installed; skipping PNG")
    return paths

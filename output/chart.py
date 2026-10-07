"""Export the four-column chart as CSV, styled HTML and PNG.

Rows are value bands (100-95, 95-90, ...), so players on the same row are
worth about the same regardless of position. Empty bands are kept so the
vertical gaps stay to scale.
"""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

POSITIONS = ["QB", "RB", "WR", "TE"]


def build_chart(values: pd.DataFrame, band: float) -> pd.DataFrame:
    """Cells are lists of 'Name (TEAM) — value', highest first."""
    v = values[values.value > 0]
    tops = np.arange(100, 0, -band)
    # Band i covers (top - band, top]; 100.0 lands in the first band.
    idx = np.clip(np.ceil((100 - v.value) / band) - 1, 0, len(tops) - 1).astype(int)
    idx[v.value >= 100] = 0
    chart = pd.DataFrame([[[] for _ in POSITIONS] for _ in tops], columns=POSITIONS,
                         index=pd.Index([f"{t:g}–{max(t - band, 0):g}" for t in tops], name="Value"))
    for (name, team, pos, val), i in zip(v[["name", "team", "position", "value"]].values, idx):
        if pos in POSITIONS:
            chart.iat[i, POSITIONS.index(pos)].append(f"{name} ({team}) — {val:.1f}")
    return chart


def _html(chart: pd.DataFrame, title: str) -> str:
    n = len(chart)
    rows = []
    for i, (label, r) in enumerate(chart.iterrows()):
        shade = f"rgba(37,99,235,{0.55 * (1 - i / n):.2f})"
        cells = []
        for players in r:
            items = "".join(
                f'<div><span class="n">{html.escape(p.rsplit(" — ", 1)[0])}</span>'
                f'<span class="v">{p.rsplit(" — ", 1)[1]}</span></div>' for p in players)
            cells.append(f'<td style="background:{shade if players else "transparent"}">{items}</td>')
        rows.append(f"<tr><th>{label}</th>{''.join(cells)}</tr>")
    head = "".join(f"<th>{p}</th>" for p in chart.columns)
    return f"""<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font:14px system-ui,sans-serif;margin:24px;background:#fafafa;color:#111}}
table{{border-collapse:collapse;width:100%;max-width:1200px;table-layout:fixed}}
th,td{{border:1px solid #ddd;padding:4px 10px;text-align:left;vertical-align:top}}
thead th{{background:#111;color:#fff;position:sticky;top:0}}
tbody th{{background:#f0f0f0;width:5.5em;text-align:right;white-space:nowrap}}
td div{{display:flex;justify-content:space-between;gap:12px;white-space:nowrap}}
.v{{font-weight:600}}
</style>
<h1>{html.escape(title)}</h1>
<table><thead><tr><th>Value</th>{head}</tr></thead><tbody>
{chr(10).join(rows)}
</tbody></table>"""


def _png(chart: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    text = [["\n".join(c) for c in r] for r in chart.values]
    lines = [max(1, *(len(c) for c in r)) for r in chart.values]
    fig, ax = plt.subplots(figsize=(18, 0.17 * sum(lines) + 1.5))
    ax.axis("off")
    t = ax.table(cellText=text, colLabels=list(chart.columns), rowLabels=list(chart.index),
                 loc="upper center", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    unit = 1 / (sum(lines) + 1)
    for (r, c), cell in t.get_celld().items():
        cell.set_height(unit * (lines[r - 1] if r > 0 else 1))
        cell.get_text().set_va("center")
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export(values: pd.DataFrame, out_dir: Path, band: float, title: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    chart = build_chart(values, band)
    paths = [out_dir / "trade_value_chart.csv", out_dir / "trade_value_chart.html",
             out_dir / "values_detail.csv"]
    chart.map(" | ".join).to_csv(paths[0])
    paths[1].write_text(_html(chart, title))
    values.round(3).to_csv(paths[2], index_label="gsis_id")
    paths.append(out_dir / "trade_value_chart.png")
    _png(chart, paths[-1], title)
    return paths

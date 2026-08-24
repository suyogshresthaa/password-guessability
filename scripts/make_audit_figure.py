"""Figure 0 for docs/00-audit-v1.md — the label is a pure function of length.

Renders docs/figures/00_label_vs_length{,-dark}.svg.

Palette: ordinal blue ramp from the data-viz reference palette (light steps
250/450/650, dark steps 200/400/600). Ordinal rather than categorical because
Weak < Medium < Strong is genuinely ordered; the old red/yellow/green scheme in
v1's app.py is a colour-vision-deficiency failure and is not reused here.
"""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "eval" / "000webhost_100k.sqlite"
OUT = ROOT / "docs" / "figures"

LABELS = ["Weak", "Medium", "Strong"]
XMAX = 24  # lengths 25..220 exist but are a negligible, all-Strong tail

THEMES = {
    "light": dict(
        surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e", muted="#8a8880",
        grid="#e6e5e1", series=["#86b6ef", "#2a78d6", "#104281"], suffix="",
    ),
    "dark": dict(
        surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7", muted="#8a8880",
        grid="#33332f", series=["#9ec5f4", "#3987e5", "#184f95"], suffix="-dark",
    ),
}


def load_counts():
    """counts[label][length] over the whole table, plus the >XMAX tail."""
    with sqlite3.connect(DB) as conn:
        rows = conn.execute(
            "select length(password), strength, count(*) from Users group by 1, 2"
        ).fetchall()
    counts = np.zeros((3, XMAX + 1), dtype=int)
    tail = np.zeros(3, dtype=int)
    for length, strength, n in rows:
        if length <= XMAX:
            counts[strength, length] += n
        else:
            tail[strength] += n
    return counts, tail


def render(theme_name, counts, tail):
    t = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    x = np.arange(1, XMAX + 1)
    ranges = ["1–7", "8–13", "≥14"]
    for cls in range(3):
        y = counts[cls, 1:]
        ax.bar(
            x[y > 0], y[y > 0],
            width=0.78, color=t["series"][cls],
            label=f"{LABELS[cls]}  ·  length {ranges[cls]}",
            linewidth=0, zorder=3,
        )

    # Interval boundaries — the whole point of the figure.
    for boundary in (7.5, 13.5):
        ax.axvline(boundary, color=t["secondary"], lw=1.1, ls=(0, (4, 3)), zorder=4)

    ax.set_xlim(0.3, XMAX + 0.7)
    ax.set_ylim(0, counts.sum(axis=0).max() * 1.32)
    ax.set_xticks(range(2, XMAX + 1, 2))
    ax.set_xlabel("Password length (characters)", color=t["secondary"], fontsize=10)
    ax.set_ylabel("Passwords", color=t["secondary"], fontsize=10)

    ax.yaxis.grid(True, color=t["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["secondary"], labelsize=9, length=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))

    # Legend plus direct labels — identity is never colour-alone.
    legend = ax.legend(
        loc="upper left", frameon=False, fontsize=9.5,
        handlelength=0.9, handleheight=0.9, borderpad=0, labelspacing=0.55,
    )
    for text in legend.get_texts():
        text.set_color(t["secondary"])

    head = ax.get_ylim()[1]
    for cls in range(3):
        peak_len = int(counts[cls].argmax())
        ax.text(
            peak_len, counts[cls, peak_len] + head * 0.028, LABELS[cls],
            ha="center", va="bottom", color=t["primary"],
            fontsize=10.5, fontweight="bold",
        )

    ax.text(
        XMAX + 0.4, head * 0.30,
        f"+{tail.sum():,} passwords of length 25–220,\nevery one labelled Strong",
        ha="right", va="center", color=t["muted"], fontsize=8.5, style="italic",
    )

    ax.set_title(
        "The strength label is exactly a length threshold",
        color=t["primary"], fontsize=14, fontweight="bold", loc="left", pad=30,
    )
    ax.text(
        0, 1.025,
        "100,000 rows · 0 exceptions · no bar is ever split between two labels",
        transform=ax.transAxes, color=t["secondary"], fontsize=9.5, va="bottom",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"00_label_vs_length{t['suffix']}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor=t["surface"])
    plt.close(fig)
    return path


if __name__ == "__main__":
    counts, tail = load_counts()
    # Guard: the figure's claim must hold, or the figure is a lie.
    for cls, (lo, hi) in enumerate([(1, 7), (8, 13), (14, XMAX)]):
        outside = counts[cls].sum() - counts[cls, lo : hi + 1].sum()
        assert outside == 0, f"class {cls} has {outside} rows outside {lo}-{hi}"
    assert tail[0] == 0 and tail[1] == 0, "tail beyond XMAX is not all-Strong"
    for path in (render("light", counts, tail), render("dark", counts, tail)):
        print(f"wrote {path.relative_to(ROOT)}")

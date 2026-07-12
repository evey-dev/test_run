"""Render a schematic showing the panel swap intervention mechanism.

Shows the clean forward pass versus the force-source swap, with logit
values drawn from the actual confirmation results. Produces a figure
suitable for inclusion in the report.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Tuple

cache_root = Path(tempfile.gettempdir()) / "mphil-project-matplotlib"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "config"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "cache"))

import matplotlib as mpl

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Colours
INK = "#17212B"
MUTED = "#5F6B76"
GRID = "#D8DEE6"
FORCE = "#C53B4C"
MASS = "#2F6FA3"
ENERGY = "#17847A"
PURPLE = "#8D5A9A"
GOLD = "#C58A13"
INPUT_BLUE = "#4C78A8"


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    face: str,
    edge: str,
    lw: float = 1.2,
    ls: str = "solid",
    zorder: int = 2,
):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        facecolor=face, edgecolor=edge, linewidth=lw, linestyle=ls,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_arrow(ax, start: Tuple, end: Tuple, color: str, lw: float = 1.5):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color, zorder=3,
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(11.0, 5.2), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(0.50, 0.96, "Intervention mechanism: force-source swap into energy target",
            fontsize=14, weight="bold", color=INK, ha="center", va="top")
    ax.text(0.50, 0.91, "Same prompt and model; only 10 panel latents at layer 28 are replaced",
            fontsize=9.5, color=MUTED, ha="center", va="top")

    # ========== ROW 1: Clean forward pass ==========
    row1_y = 0.62
    ax.text(0.03, row1_y + 0.14, "(a) Clean forward pass", fontsize=10.5, weight="bold", color=INK)

    # Input prompt
    add_box(ax, 0.03, row1_y - 0.05, 0.17, 0.12, "#E7F0F8", INPUT_BLUE)
    ax.text(0.115, row1_y + 0.035, '"...energy\nis named "', fontsize=8.2, color=INK,
            ha="center", va="center", family="monospace", linespacing=1.2)

    # Model block
    add_box(ax, 0.25, row1_y - 0.07, 0.32, 0.16, "#F8F9FA", GRID, lw=1.0)
    ax.text(0.41, row1_y + 0.075, "Transformer + MLP (layers 4–28)", fontsize=8, color=MUTED, ha="center")

    # Panel inside model
    add_box(ax, 0.33, row1_y - 0.04, 0.16, 0.09, "#F0E7F2", PURPLE, ls=(0, (3, 2)))
    ax.text(0.41, row1_y + 0.005, "10 panel latents\n(L28)", fontsize=7.5, color=PURPLE,
            ha="center", va="center", linespacing=1.15)

    # Output logits
    add_box(ax, 0.62, row1_y - 0.05, 0.18, 0.12, "#FBF1D8", GOLD)
    ax.text(0.71, row1_y + 0.045, r"$\ell$(new) $-$ $\ell$(j)", fontsize=8.5, color=INK,
            ha="center", va="center")
    ax.text(0.71, row1_y - 0.015, "= $-9.41$", fontsize=9, color=INK, ha="center", va="center", weight="bold")

    # Prediction
    add_box(ax, 0.84, row1_y - 0.03, 0.13, 0.08, "#E4F3F0", ENERGY)
    ax.text(0.905, row1_y + 0.01, "Top: joules", fontsize=8.5, color=ENERGY,
            ha="center", va="center", weight="bold")

    # Arrows
    add_arrow(ax, (0.20, row1_y + 0.01), (0.25, row1_y + 0.01), INPUT_BLUE)
    add_arrow(ax, (0.57, row1_y + 0.01), (0.62, row1_y + 0.01), MUTED)
    add_arrow(ax, (0.80, row1_y + 0.01), (0.84, row1_y + 0.01), GOLD)

    # ========== ROW 2: Force-source swap ==========
    row2_y = 0.14
    row2_heading_y = 0.39
    ax.text(0.03, row2_heading_y, "(b) After force-source swap into panel", fontsize=10.5, weight="bold", color=INK)

    # Input prompt (same)
    add_box(ax, 0.03, row2_y - 0.05, 0.17, 0.12, "#E7F0F8", INPUT_BLUE)
    ax.text(0.115, row2_y + 0.035, '"...energy\nis named "', fontsize=8.2, color=INK,
            ha="center", va="center", family="monospace", linespacing=1.2)

    # Model block
    add_box(ax, 0.25, row2_y - 0.07, 0.32, 0.16, "#F8F9FA", GRID, lw=1.0)
    ax.text(0.41, row2_y + 0.075, "Transformer + MLP (layers 4–28)", fontsize=8, color=MUTED, ha="center")

    # Panel with FORCE swap
    add_box(ax, 0.33, row2_y - 0.04, 0.16, 0.09, "#F9E7EA", FORCE)
    ax.text(0.41, row2_y + 0.005, "force values\nswapped in", fontsize=7.5, color=FORCE,
            ha="center", va="center", weight="bold", linespacing=1.15)

    # Force donor arrow
    add_box(ax, 0.35, row2_y + 0.14, 0.12, 0.06, "#F9E7EA", FORCE)
    ax.text(0.41, row2_y + 0.17, "Force donor", fontsize=7.5, color=FORCE, ha="center", va="center")
    add_arrow(ax, (0.41, row2_y + 0.14), (0.41, row2_y + 0.055), FORCE)

    # Output logits
    add_box(ax, 0.62, row2_y - 0.05, 0.18, 0.12, "#FBF1D8", GOLD)
    ax.text(0.71, row2_y + 0.045, r"$\ell$(new) $-$ $\ell$(j)", fontsize=8.5, color=INK,
            ha="center", va="center")
    ax.text(0.71, row2_y - 0.015, "= $-8.17$", fontsize=9, color=INK, ha="center", va="center", weight="bold")

    # Prediction (same top but shifted)
    add_box(ax, 0.84, row2_y - 0.03, 0.13, 0.08, "#E4F3F0", ENERGY)
    ax.text(0.905, row2_y + 0.01, "Top: joules", fontsize=8.5, color=ENERGY,
            ha="center", va="center", weight="bold")

    # Delta annotation
    ax.text(0.905, row2_y - 0.06, r"$\Delta G = +1.23$ logits", fontsize=8.5, color=FORCE,
            ha="center", va="center", weight="bold")
    ax.text(0.905, row2_y - 0.09, "(toward newtons)", fontsize=7.5, color=MUTED, ha="center")

    # Arrows
    add_arrow(ax, (0.20, row2_y + 0.01), (0.25, row2_y + 0.01), INPUT_BLUE)
    add_arrow(ax, (0.57, row2_y + 0.01), (0.62, row2_y + 0.01), MUTED)
    add_arrow(ax, (0.80, row2_y + 0.01), (0.84, row2_y + 0.01), GOLD)

    # Bottom note
    ax.text(0.50, 0.02,
            "The 10 frozen panel features shift the logit gap by +1.23 on average across 16 systems. "
            "A matched mass-source control shifts it by $-$0.09, confirming force specificity.",
            fontsize=8, color=MUTED, ha="center", va="center")

    output_dir = Path("report/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fig_intervention_schematic.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig_intervention_schematic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_dir / 'fig_intervention_schematic.pdf'}")


if __name__ == "__main__":
    main()

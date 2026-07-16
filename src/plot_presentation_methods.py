"""Render slide-native method diagrams for the oral presentation."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence


INK = "#17212B"
MUTED = "#5F6B76"
GRID = "#D8DEE6"
BLUE = "#2F6FA3"
BLUE_LIGHT = "#E7F0F8"
TEAL = "#17847A"
TEAL_LIGHT = "#E4F3F0"
PURPLE = "#7B5AA6"
PURPLE_LIGHT = "#F0EAF6"
RED = "#C53B4C"
RED_LIGHT = "#F9E7EA"
GOLD = "#B7791F"
GOLD_LIGHT = "#FFF3D6"
GREY_LIGHT = "#F6F8FA"


def configure_matplotlib() -> None:
    root = Path(tempfile.gettempdir()) / "mphil-project-matplotlib"
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(root / "config"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / "cache"))
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def rounded_box(axis, x, y, width, height, face, edge, radius=0.012, linewidth=1.3):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
    )
    axis.add_patch(patch)
    return patch


def arrow(axis, start, end, colour=INK, width=1.5, style="-|>", dashed=False):
    from matplotlib.patches import FancyArrowPatch

    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=width,
        linestyle=(0, (4, 3)) if dashed else "solid",
        color=colour,
        shrinkA=1,
        shrinkB=1,
    )
    axis.add_patch(patch)
    return patch


def base_figure(title: str, subtitle: str):
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16, 9), facecolor="white")
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0.055, 0.925, title, fontsize=22, weight="bold", color=INK, va="top")
    axis.text(0.055, 0.875, subtitle, fontsize=11, color=MUTED, va="top")
    return figure, axis


def save_figure(figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.{suffix}" for suffix in ("png", "pdf", "svg")]
    for path in paths:
        kwargs = {"bbox_inches": "tight", "facecolor": figure.get_facecolor()}
        if path.suffix == ".png":
            kwargs["dpi"] = 240
        figure.savefig(path, **kwargs)
    return paths


def render_pipeline(output_dir: Path) -> list[Path]:
    figure, axis = base_figure(
        "From prompt corpus to a controlled mechanistic claim",
        "Independent implementation on Qwen3-4B-Instruct-2507 | fixed seed 787 | selected layers 4, 8, 12, 16, 20, 24, 28",
    )
    stages = [
        ("PROMPTS", "1,000 original\n10,000 targeted extension", BLUE, BLUE_LIGHT),
        ("ACTIVATIONS", "final-token MLP state\nat seven layers", TEAL, TEAL_LIGHT),
        ("TOPK SAE", "8,192 learned directions\nk = 128 or 256", PURPLE, PURPLE_LIGHT),
        ("CONTRAST GRAPH", "$J = \\ell_t - \\ell_c$\npruned candidate set", GOLD, GOLD_LIGHT),
        ("CAUSAL TEST", "inhibit or swap\nagainst matched control", RED, RED_LIGHT),
    ]
    x_values = [0.055, 0.245, 0.435, 0.625, 0.815]
    for index, ((heading, body, edge, face), x) in enumerate(zip(stages, x_values)):
        rounded_box(axis, x, 0.590, 0.135, 0.165, face, edge)
        axis.text(x + 0.0675, 0.708, heading, fontsize=8.2, color=edge, weight="bold", ha="center")
        axis.text(x + 0.0675, 0.650, body, fontsize=9.4, color=INK, weight="bold", ha="center", va="center", linespacing=1.45)
        if index < len(stages) - 1:
            arrow(axis, (x + 0.137, 0.672), (x_values[index + 1] - 0.004, 0.672), colour=MUTED)

    axis.text(0.055, 0.525, "THE INFERENTIAL DISCIPLINE", fontsize=8.5, color=MUTED, weight="bold")
    lower = [
        (0.090, "DISCOVERY", "rank candidate features\non designated cases", BLUE, BLUE_LIGHT),
        (0.355, "FREEZE", "fix feature IDs, panel size,\ncontrast and success rule", PURPLE, PURPLE_LIGHT),
        (0.620, "CONFIRMATION", "evaluate once on disjoint\ncases and matched controls", TEAL, TEAL_LIGHT),
        (0.845, "CLAIM GATE", "effect direction +\nbootstrap interval", RED, RED_LIGHT),
    ]
    for index, (x, heading, body, edge, face) in enumerate(lower):
        width = 0.170 if index < 3 else 0.105
        rounded_box(axis, x, 0.300, width, 0.145, face, edge)
        axis.text(x + width / 2, 0.405, heading, fontsize=8.2, color=edge, weight="bold", ha="center")
        axis.text(x + width / 2, 0.350, body, fontsize=8.7, color=INK, weight="bold", ha="center", va="center", linespacing=1.4)
        if index < len(lower) - 1:
            next_x = lower[index + 1][0]
            arrow(axis, (x + width + 0.002, 0.372), (next_x - 0.004, 0.372), colour=MUTED)
    arrow(axis, (0.882, 0.590), (0.897, 0.447), colour=RED, width=1.7)

    rounded_box(axis, 0.055, 0.115, 0.895, 0.095, GREY_LIGHT, GRID)
    axis.text(0.075, 0.171, "POST-REPORT SCALE EXTENSION", fontsize=8.2, color=INK, weight="bold")
    axis.text(
        0.075,
        0.137,
        "Units: 10k physics prompts  |  Mathematics: 10k tens-position prompts  |  Capitals: 10k capital/location relation prompts",
        fontsize=9.4,
        color=MUTED,
        weight="bold",
    )
    return save_figure(figure, output_dir, "fig_presentation_reproduction_pipeline")


def render_sae(output_dir: Path) -> list[Path]:
    figure, axis = base_figure(
        "The sparse representation and the edit actually applied",
        "TopK controls support size directly; error-preserving edits change selected decoder contributions without replacing the full MLP activation",
    )

    rounded_box(axis, 0.055, 0.530, 0.135, 0.180, BLUE_LIGHT, BLUE)
    axis.text(0.1225, 0.675, "MLP OUTPUT", fontsize=8.2, color=BLUE, weight="bold", ha="center")
    axis.text(0.1225, 0.615, "$x_\\ell \\in \\mathbb{R}^{2560}$", fontsize=16, color=INK, ha="center")
    axis.text(0.1225, 0.558, "dense activation", fontsize=8.5, color=MUTED, ha="center")

    rounded_box(axis, 0.255, 0.530, 0.155, 0.180, TEAL_LIGHT, TEAL)
    axis.text(0.3325, 0.675, "ENCODER", fontsize=8.2, color=TEAL, weight="bold", ha="center")
    axis.text(0.3325, 0.620, "$a = \\mathrm{ReLU}(W_e(u-b_d)+b_e)$", fontsize=11.5, color=INK, ha="center")
    axis.text(0.3325, 0.568, "$u=x_\\ell/s_\\ell$", fontsize=10.2, color=MUTED, ha="center")

    rounded_box(axis, 0.475, 0.500, 0.185, 0.240, PURPLE_LIGHT, PURPLE)
    axis.text(0.5675, 0.700, "SPARSE CODE", fontsize=8.2, color=PURPLE, weight="bold", ha="center")
    axis.text(0.5675, 0.662, "$z=\\operatorname{TopK}_k(a)$", fontsize=14, color=INK, ha="center")
    bar_x = [0.508 + index * 0.012 for index in range(10)]
    heights = [0.015, 0.090, 0.012, 0.045, 0.014, 0.112, 0.010, 0.072, 0.013, 0.030]
    for x, height in zip(bar_x, heights):
        axis.add_patch(__import__("matplotlib").patches.Rectangle((x, 0.545), 0.007, height, facecolor=PURPLE if height > 0.025 else GRID, edgecolor="none"))
    axis.text(0.5675, 0.518, "$z \\in \\mathbb{R}^{8192}$; at most $k$ active", fontsize=8.7, color=MUTED, ha="center")

    rounded_box(axis, 0.725, 0.530, 0.205, 0.180, GOLD_LIGHT, GOLD)
    axis.text(0.8275, 0.675, "DECODER", fontsize=8.2, color=GOLD, weight="bold", ha="center")
    axis.text(0.8275, 0.620, "$\\hat{u}=W_dz+b_d$", fontsize=14, color=INK, ha="center")
    axis.text(0.8275, 0.568, "$\\|W_{d,:,j}\\|_2=1$", fontsize=10.2, color=MUTED, ha="center")
    arrow(axis, (0.192, 0.620), (0.252, 0.620), colour=MUTED)
    arrow(axis, (0.412, 0.620), (0.472, 0.620), colour=MUTED)
    arrow(axis, (0.662, 0.620), (0.722, 0.620), colour=MUTED)

    axis.text(0.055, 0.425, "ERROR-PRESERVING CAUSAL EDITS", fontsize=8.5, color=MUTED, weight="bold")
    rounded_box(axis, 0.055, 0.185, 0.405, 0.185, RED_LIGHT, RED)
    axis.text(0.075, 0.327, "INHIBIT A FROZEN FEATURE SET $S$", fontsize=8.2, color=RED, weight="bold")
    axis.text(0.2575, 0.275, "$x'_\\ell = x_\\ell - s_\\ell W_{d,S}z_{\\ell,S}$", fontsize=15, color=INK, ha="center")
    axis.text(0.2575, 0.220, "subtract only the selected decoder contribution", fontsize=8.8, color=MUTED, ha="center")

    rounded_box(axis, 0.540, 0.185, 0.390, 0.185, BLUE_LIGHT, BLUE)
    axis.text(0.560, 0.327, "SWAP THE SAME SET FROM A DONOR", fontsize=8.2, color=BLUE, weight="bold")
    axis.text(0.735, 0.275, "$x'_\\ell=x_\\ell+s_\\ell W_{d,S}(z^{src}_{\\ell,S}-z^{tgt}_{\\ell,S})$", fontsize=13.0, color=INK, ha="center")
    axis.text(0.735, 0.220, "retain the target reconstruction error and all unedited directions", fontsize=8.8, color=MUTED, ha="center")
    return save_figure(figure, output_dir, "fig_presentation_sae_and_intervention")


def render_evidence(output_dir: Path) -> list[Path]:
    figure, axis = base_figure(
        "Three evidential questions, three different measurements",
        "A faithful reconstruction or a decodable signal does not by itself identify a sparse causal mechanism",
    )
    columns = [
        (0.055, "1", "RECONSTRUCTION", "$\\mathrm{FVE}=1-\\frac{\\sum_i\\|x_i-\\hat{x}_i\\|^2}{\\sum_i\\|x_i-\\bar{x}\\|^2}$", "Does the SAE preserve clean activations?", BLUE, BLUE_LIGHT),
        (0.365, "2", "DECODABILITY", "$\\mathrm{AUC}(q^\\top x, y)$", "Can a classifier recover the variable?", PURPLE, PURPLE_LIGHT),
        (0.675, "3", "CAUSAL SPECIFICITY", "$\\Delta_{pair}=\\Delta_{target}-\\Delta_{control}$", "Does a frozen edit move the intended contrast selectively?", RED, RED_LIGHT),
    ]
    for x, number, heading, formula, question, edge, face in columns:
        rounded_box(axis, x, 0.430, 0.270, 0.335, face, edge)
        axis.text(x + 0.025, 0.722, number, fontsize=10, color="white", weight="bold", ha="center", va="center", bbox={"boxstyle": "circle,pad=0.35", "facecolor": edge, "edgecolor": edge})
        axis.text(x + 0.055, 0.722, heading, fontsize=9.0, color=edge, weight="bold", va="center")
        axis.text(x + 0.135, 0.620, formula, fontsize=13.0, color=INK, ha="center")
        axis.text(x + 0.135, 0.535, question, fontsize=9.5, color=INK, weight="bold", ha="center", va="center", wrap=True)
        axis.text(x + 0.135, 0.470, ["ReLU SAEs:\nhigh FVE, dense codes", "Balanced carry probe:\nAUC 1.0, Top-10 causal null", "10k units panel:\nheld-out paired shift"][int(number) - 1], fontsize=8.4, color=MUTED, ha="center", va="center")
    arrow(axis, (0.327, 0.598), (0.360, 0.598), colour=MUTED, dashed=True)
    arrow(axis, (0.637, 0.598), (0.670, 0.598), colour=MUTED, dashed=True)
    axis.text(0.346, 0.622, "$\\nRightarrow$", fontsize=17, color=RED, weight="bold", ha="center")
    axis.text(0.656, 0.622, "$\\nRightarrow$", fontsize=17, color=RED, weight="bold", ha="center")

    rounded_box(axis, 0.175, 0.180, 0.650, 0.135, GREY_LIGHT, GRID)
    axis.text(0.500, 0.270, "MECHANISTIC CLAIM STANDARD", fontsize=8.5, color=INK, weight="bold", ha="center")
    axis.text(0.500, 0.225, "Frozen feature set + disjoint confirmation cases + matched control + uncertainty interval", fontsize=11.0, color=INK, weight="bold", ha="center")
    return save_figure(figure, output_dir, "fig_presentation_evidence_standard")


def render_baselines(output_dir: Path) -> list[Path]:
    figure, axis = base_figure(
        "Baseline competence before intervention",
        "Expected answer prefix at rank 1 or anywhere in the recorded Top-20 next-token distribution",
    )
    import numpy as np

    chart = figure.add_axes([0.11, 0.22, 0.80, 0.53])
    labels = ["Multi-step\naddition", "Physics-based\nSI units", "Multi-step\ncapital retrieval"]
    top1 = np.asarray([100.0, 67.0, 49.7354497354])
    top20 = np.asarray([100.0, 96.5, 83.5978835979])
    x = np.arange(len(labels))
    width = 0.29
    bars1 = chart.bar(x - width / 2, top1, width, color=BLUE, label="Top-1")
    bars2 = chart.bar(x + width / 2, top20, width, color=TEAL, label="In Top-20")
    chart.set_ylim(0, 110)
    chart.set_ylabel("Prompts with expected answer prefix (%)", color=INK, fontsize=10)
    chart.set_xticks(x, labels, fontsize=10, weight="bold")
    chart.set_yticks([0, 25, 50, 75, 100])
    chart.grid(axis="y", color=GRID, linewidth=0.8)
    chart.set_axisbelow(True)
    chart.spines[["top", "right"]].set_visible(False)
    chart.spines[["left", "bottom"]].set_color(MUTED)
    chart.legend(frameon=False, loc="upper right", ncols=2)
    for bars in (bars1, bars2):
        for bar in bars:
            chart.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{bar.get_height():.1f}%" if bar.get_height() != 100 else "100%",
                ha="center",
                va="bottom",
                color=INK,
                fontsize=9,
                weight="bold",
            )
    axis.text(
        0.50,
        0.130,
        "Evaluation pools: addition n=200 | SI units n=200 | capitals n=189",
        fontsize=9.2,
        color=MUTED,
        ha="center",
    )
    axis.text(
        0.50,
        0.090,
        "Every causal benchmark applies its own clean-baseline eligibility filter before intervention.",
        fontsize=9.2,
        color=INK,
        weight="bold",
        ha="center",
    )
    return save_figure(figure, output_dir, "fig_presentation_baseline_competence")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render oral-presentation method diagrams")
    parser.add_argument("--output-dir", type=Path, default=Path("presentation/figures"))
    args = parser.parse_args()
    configure_matplotlib()
    paths: Iterable[Path] = (
        render_pipeline(args.output_dir)
        + render_sae(args.output_dir)
        + render_evidence(args.output_dir)
        + render_baselines(args.output_dir)
    )
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

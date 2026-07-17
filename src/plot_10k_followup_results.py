"""Create presentation figures for the 10,000-prompt follow-up experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


INK = "#17212B"
MUTED = "#5F6B76"
GRID = "#D8DEE4"
SOFT = "#F4F6F8"
BLUE = "#2F6FA3"
BLUE_LIGHT = "#DCEAF5"
TEAL = "#148A87"
TEAL_LIGHT = "#D9F0ED"
RED = "#C64B45"
RED_LIGHT = "#F6E1DF"
GOLD = "#B88416"
GOLD_LIGHT = "#F7EDCF"
PURPLE = "#7454A5"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rounded_box(
    axis: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str,
    edge: str,
    radius: float = 0.012,
    linewidth: float = 1.1,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            transform=axis.transAxes,
            boxstyle=f"round,pad=0.008,rounding_size={radius}",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
        )
    )


def arrow(
    axis: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    linewidth: float = 1.5,
    style: str = "-|>",
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=axis.transAxes,
            arrowstyle=style,
            mutation_scale=11,
            linewidth=linewidth,
            color=color,
        )
    )


def format_axis(axis: Axes, *, grid_axis: str = "x") -> None:
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
    axis.tick_params(axis="both", colors=MUTED, labelsize=8)


def forest_plot(
    axis: Axes,
    rows: Sequence[tuple[str, float, Sequence[float], str]],
    *,
    xlabel: str,
    xlim: tuple[float, float],
) -> None:
    y_positions = np.arange(len(rows))[::-1]
    axis.axvline(0, color=INK, linewidth=1.0, zorder=1)
    for y, (label, mean, interval, color) in zip(y_positions, rows):
        axis.plot(interval, [y, y], color=color, linewidth=3.0, solid_capstyle="round", zorder=2)
        axis.scatter([mean], [y], s=66, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        axis.text(interval[1] + (xlim[1] - xlim[0]) * 0.025, y, f"{mean:+.3f}", color=color,
                  fontsize=8.5, weight="bold", va="center")
    axis.set_yticks(y_positions, [row[0] for row in rows])
    axis.set_xlim(*xlim)
    axis.set_ylim(-0.6, len(rows) - 0.4)
    axis.set_xlabel(xlabel, fontsize=8.2, color=MUTED)
    format_axis(axis)
    axis.tick_params(axis="y", length=0, colors=INK, labelsize=8.5)


def save_figure(figure: Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        path = output_dir / f"{stem}.{suffix}"
        kwargs = {"dpi": 260} if suffix == "png" else {}
        figure.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(figure)


def panel_by_name(payload: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next(panel for panel in payload["confirmation"]["panels"] if panel["name"] == name)


def causal_panel_by_name(payload: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next(panel for panel in payload["causal_confirmation"]["panels"] if panel["name"] == name)


def plot_math_graph_screen(
    payload: Dict[str, Any],
    output_dir: Path,
    *,
    show_all_random_controls: bool = True,
    stem: str = "fig_math_10k_graph_carry_panel",
) -> None:
    primary = payload["confirmation"]["primary_result"]
    summary = primary["summary"]
    top10 = panel_by_name(payload, "top_10_primary")
    top1 = panel_by_name(payload, "top_1")["summary"]

    figure = plt.figure(figsize=(12.0, 6.75), facecolor="white")
    canvas = figure.add_axes([0, 0, 1, 1])
    canvas.axis("off")
    canvas.text(0.045, 0.93, "A graph-seeded carry panel passes held-out specificity",
                fontsize=18, weight="bold", color=INK)
    canvas.text(0.045, 0.888,
                "10,000-prompt TopK-256 SAEs | final-token inhibition | 24 disjoint confirmation pairs",
                fontsize=9.2, color=MUTED)

    canvas.text(0.055, 0.818, "A", fontsize=10, color="white", weight="bold", ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.28", "facecolor": INK, "edgecolor": INK})
    canvas.text(0.082, 0.818, "Graph seed and discovery ranking", fontsize=12.5,
                color=INK, weight="bold", va="center")

    rounded_box(canvas, 0.055, 0.645, 0.155, 0.105, face=RED_LIGHT, edge=RED)
    canvas.text(0.1325, 0.718, "ATTRIBUTION GRAPH SEED", fontsize=7.3,
                color=RED, weight="bold", ha="center")
    canvas.text(0.1325, 0.680, "58 + 83 = 141", fontsize=11, color=INK, weight="bold", ha="center")
    canvas.text(0.1325, 0.654, "score: logit(4) - logit(3)", fontsize=7.1, color=MUTED, ha="center")

    rounded_box(canvas, 0.055, 0.475, 0.155, 0.105, face=BLUE_LIGHT, edge=BLUE)
    canvas.text(0.1325, 0.56, "DISCOVERY SCREEN", fontsize=7.3,
                color=BLUE, weight="bold", ha="center")
    canvas.text(0.1325, 0.535, "matched pairs", fontsize=10.5, color=INK, weight="bold", ha="center")
    canvas.text(0.1325, 0.510, "67 + 75 = 142, 64 + 75 = 139", fontsize=8, color=MUTED, weight="bold", ha="center")
    canvas.text(0.1325, 0.484, "rank 76 graph candidates", fontsize=7.1,
                color=MUTED, ha="center")

    rounded_box(canvas, 0.258, 0.475, 0.195, 0.275, face=SOFT, edge=GRID)
    canvas.text(0.3555, 0.718, "FROZEN TOP-10 PANEL", fontsize=8, color=PURPLE, weight="bold", ha="center")
    feature_rows: Iterable[str] = (
        "L24  F2070  F6132  F1509",
        "L20  F7889",
        "L12  F7303  F1275  F7728",
        "L8    F1922",
        "L28  F750   F1204",
    )
    for index, text in enumerate(feature_rows):
        y = 0.675 - index * 0.038
        canvas.text(0.278, y, text, fontsize=7.6, color=INK,
                    family="monospace", weight="bold" if index == 0 else "normal")
    canvas.text(0.278, 0.494, "Freeze before confirmation", fontsize=7.1, color=MUTED)
    canvas.text(0.278, 0.473, "and inhibit the same coordinates", fontsize=7.1, color=MUTED)
    arrow(canvas, (0.1325, 0.640), (0.1325, 0.585), color=RED)
    arrow(canvas, (0.210, 0.527), (0.250, 0.565), color=BLUE)

    rounded_box(canvas, 0.055, 0.310, 0.398, 0.090, face=GOLD_LIGHT, edge=GOLD)
    canvas.text(0.073, 0.365, "Single-feature cross-check", fontsize=8.3, color=GOLD, weight="bold")
    canvas.text(0.073, 0.335,
                f"L24F2070 active in {top1['target_active_fraction'] * 24:.0f}/24 carry targets and "
                f"{top1['control_active_fraction'] * 24:.0f}/24 controls; paired effect {top1['mean_paired_difference']:+.3f}",
                fontsize=8.0, color=INK)

    canvas.text(0.535, 0.818, "B", fontsize=10, color="white", weight="bold", ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.28", "facecolor": INK, "edgecolor": INK})
    canvas.text(0.562, 0.818, "Primary confirmation effect", fontsize=12.5, color=INK, weight="bold", va="center")
    forest = figure.add_axes([0.58, 0.53, 0.355, 0.235])
    forest_plot(
        forest,
        [
            ("Carry target", summary["mean_target_delta"],
             summary["bootstrap_95_ci_mean_target_delta"], RED),
            ("No-carry control", summary["mean_no_carry_control_delta"],
             summary["bootstrap_95_ci_mean_no_carry_control_delta"], BLUE),
            ("Paired specificity", summary["mean_paired_difference"],
             summary["bootstrap_95_ci_mean_paired_difference"], PURPLE),
        ],
        xlabel="Change in target-minus-contrast logit gap",
        xlim=(-0.48, 0.12),
    )

    canvas.text(0.535, 0.435, "C", fontsize=10, color="white", weight="bold", ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.28", "facecolor": INK, "edgecolor": INK})
    controls_title = "Compactness and all random controls" if show_all_random_controls else "Compactness and matched controls"
    canvas.text(0.562, 0.435, controls_title, fontsize=12.5, color=INK, weight="bold", va="center")
    curve_axis = figure.add_axes([0.58, 0.16, 0.355, 0.225])
    names = ["top_1", "top_3", "top_5", "top_10_primary", "top_20", "all_positive_graph"]
    sizes = [1, 3, 5, 10, 20, payload["candidate_feature_count"]]
    values = [panel_by_name(payload, name)["summary"]["mean_paired_difference"] for name in names]
    curve_axis.plot(range(len(sizes)), values, color=PURPLE, linewidth=2.2, marker="o", markersize=6)
    curve_axis.axhline(0, color=INK, linewidth=0.9)
    all_random_values = primary["random_control_mean_paired_differences"]
    primary_effect = summary["mean_paired_difference"]
    weaker_random_values = [value for value in all_random_values if value >= primary_effect]
    random_values = all_random_values if show_all_random_controls else weaker_random_values
    random_label = (
        "all five layer-matched random Top-10 panels"
        if show_all_random_controls
        else "weaker matched random Top-10 panels"
    )
    curve_axis.scatter([3] * len(random_values), random_values, marker="x", s=42, color=GOLD,
                       linewidth=1.5, label=random_label, zorder=4)
    curve_axis.scatter([3], [summary["mean_paired_difference"]], s=75, facecolor="white",
                       edgecolor=RED, linewidth=2, zorder=5, label="primary Top-10")
    curve_axis.set_xticks(range(len(sizes)), [str(size) for size in sizes])
    curve_axis.set_ylabel("Paired specificity", fontsize=8.2, color=MUTED)
    curve_axis.set_xlabel("Number of graph features inhibited", fontsize=8.2, color=MUTED)
    curve_axis.set_ylim(-0.40, 0.14)
    format_axis(curve_axis, grid_axis="y")
    curve_axis.legend(frameon=False, fontsize=6.8, loc="lower left")

    rounded_box(canvas, 0.055, 0.120, 0.398, 0.095, face=TEAL_LIGHT, edge=TEAL)
    canvas.text(0.073, 0.180,
                "Primary rule passed: paired 95% CI [-0.276, -0.089]",
                fontsize=8.8, color=INK, weight="bold")
    canvas.text(0.073, 0.146,
                "Carry weakens more than no-carry; no top token changes.",
                fontsize=8.0, color=INK)
    controls_note = (
        "One of five random Top-10 panels was stronger, consistent with a distributed representation."
        if show_all_random_controls
        else "The primary outperformed four of five matched random panels; the full control set is retained in the backup figure."
    )
    canvas.text(0.055, 0.055,
                "Feature ranking used eight discovery pairs; confirmation used twenty-four disjoint pairs absent from the SAE corpus. "
                + controls_note,
                fontsize=7.5, color=MUTED)
    save_figure(figure, output_dir, stem)


def plot_math_balanced(
    localisation: Dict[str, Any],
    replication: Dict[str, Any],
    output_dir: Path,
) -> None:
    primary = localisation["primary_result"]
    activation = primary["activation_confirmation"]
    causal = primary["causal_confirmation"]
    replicated = replication["primary_result"]["causal_summary"]

    figure = plt.figure(figsize=(12.0, 6.75), facecolor="white")
    canvas = figure.add_axes([0, 0, 1, 1])
    canvas.axis("off")
    canvas.text(0.045, 0.93, "Balanced localisation independently confirms a distributed carry signal",
                fontsize=18, weight="bold", color=INK)
    canvas.text(0.045, 0.888,
                "Output-digit conditioning separates carry status from the digit being predicted",
                fontsize=9.2, color=MUTED)

    canvas.text(0.055, 0.805, "A", fontsize=10, color="white", weight="bold", ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.28", "facecolor": INK, "edgecolor": INK})
    canvas.text(0.082, 0.805, "Balanced discovery", fontsize=11.4,
                color=INK, weight="bold", va="center")
    rounded_box(canvas, 0.055, 0.530, 0.265, 0.205, face=SOFT, edge=GRID)
    canvas.text(0.1875, 0.696, "32 carry + 32 no-carry discovery pairs", fontsize=8.8, color=INK, weight="bold", ha="center")
    canvas.text(0.1875, 0.656, "centre activations within predicted tens digit", fontsize=8.1, color=MUTED, ha="center")
    arrow(canvas, (0.1875, 0.632), (0.1875, 0.592), color=PURPLE)
    canvas.text(0.1875, 0.570, "57,344 SAE latents", fontsize=9.5, color=PURPLE, weight="bold", ha="center")
    canvas.text(0.1875, 0.540, "261 pass fixed activity and stratum filters", fontsize=7.7, color=MUTED, ha="center")

    rounded_box(canvas, 0.055, 0.345, 0.265, 0.125, face=GOLD_LIGHT, edge=GOLD)
    canvas.text(0.074, 0.432, "Cross-protocol convergence", fontsize=8.5, color=GOLD, weight="bold")
    canvas.text(0.074, 0.398, "L24F2070 and L28F1204 recur in", fontsize=8.0, color=INK)
    canvas.text(0.074, 0.372, "both graph-ranked and activation-ranked Top-10 panels.", fontsize=8.0, color=INK)

    canvas.text(0.410, 0.805, "B", fontsize=10, color="white", weight="bold", ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.28", "facecolor": INK, "edgecolor": INK})
    canvas.text(0.437, 0.805, "Frozen Top-10 primary", fontsize=11.4,
                color=INK, weight="bold", va="center")
    rounded_box(canvas, 0.410, 0.670, 0.235, 0.065, face=TEAL_LIGHT, edge=TEAL)
    canvas.text(0.427, 0.711, "Confirmation AUC", fontsize=7.3, color=MUTED)
    canvas.text(0.535, 0.711, f"{activation['output_digit_conditioned_auc']:.3f}", fontsize=11, color=TEAL, weight="bold")
    canvas.text(0.427, 0.682, "standardised carry difference", fontsize=7.3, color=MUTED)
    canvas.text(0.590, 0.682, f"{activation['mean_within_digit_carry_minus_no_carry']:+.3f}", fontsize=9, color=TEAL, weight="bold")
    primary_axis = figure.add_axes([0.435, 0.385, 0.200, 0.235])
    forest_plot(
        primary_axis,
        [
            ("Carry target", causal["mean_carry_target_delta"],
             causal["bootstrap_95_ci_mean_carry_target_delta"], RED),
            ("No-carry control", causal["mean_no_carry_control_delta"],
             causal["bootstrap_95_ci_mean_no_carry_control_delta"], BLUE),
            ("Paired", causal["mean_paired_difference"],
             causal["bootstrap_95_ci_mean_paired_difference"], PURPLE),
        ],
        xlabel="Logit-gap change",
        xlim=(-0.30, 0.11),
    )

    canvas.text(0.720, 0.805, "C", fontsize=10, color="white", weight="bold", ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.28", "facecolor": INK, "edgecolor": INK})
    canvas.text(0.747, 0.805, "Frozen Top-20 replication", fontsize=11.4,
                color=INK, weight="bold", va="center")
    rounded_box(canvas, 0.720, 0.670, 0.235, 0.065, face=RED_LIGHT, edge=RED)
    canvas.text(0.737, 0.711, "32 intervention-untouched pairs", fontsize=8.0, color=INK, weight="bold")
    canvas.text(0.737, 0.682, "no reranking; one panel; one test", fontsize=7.4, color=MUTED)
    replication_axis = figure.add_axes([0.755, 0.385, 0.190, 0.235])
    forest_plot(
        replication_axis,
        [
            ("Carry target", replicated["mean_carry_target_delta"],
             replicated["bootstrap_95_ci_mean_carry_target_delta"], RED),
            ("No-carry control", replicated["mean_no_carry_control_delta"],
             replicated["bootstrap_95_ci_mean_no_carry_control_delta"], BLUE),
            ("Paired", replicated["mean_paired_difference"],
             replicated["bootstrap_95_ci_mean_paired_difference"], PURPLE),
        ],
        xlabel="Logit-gap change",
        xlim=(-0.55, 0.19),
    )

    rounded_box(canvas, 0.055, 0.185, 0.900, 0.095, face=TEAL_LIGHT, edge=TEAL)
    canvas.text(0.073, 0.245,
                "Top-10 primary: paired effect -0.137, 95% CI [-0.223, -0.059]",
                fontsize=10.2, color=INK, weight="bold")
    canvas.text(0.073, 0.209,
                "Frozen Top-20 replication: paired effect -0.332, 95% CI [-0.457, -0.207]",
                fontsize=10.2, color=INK, weight="bold")
    canvas.text(0.055, 0.105,
                "Interpretation: carry status is strongly represented and selected coordinates make a reproducible causal contribution. "
                "No intervention changed the top digit, and one random Top-10 panel was also significant.",
                fontsize=8.2, color=INK)
    canvas.text(0.055, 0.065,
                "The evidence supports a distributed carry-associated panel, not a unique or sufficient carry circuit.",
                fontsize=8.8, color=PURPLE, weight="bold")
    save_figure(figure, output_dir, "fig_math_10k_balanced_carry_replication")


def plot_capitals_relation(payload: Dict[str, Any], output_dir: Path) -> None:
    primary = payload["confirmation"]["primary_result"]
    summary = primary["summary"]

    figure = plt.figure(figsize=(12.0, 6.75), facecolor="white")
    canvas = figure.add_axes([0, 0, 1, 1])
    canvas.axis("off")
    canvas.text(0.045, 0.93, "A three-feature panel selectively supports capital retrieval",
                fontsize=18, weight="bold", color=INK)
    canvas.text(0.045, 0.888,
                "10,000 relation-balanced prompts | TopK-128 SAEs | 16 disjoint confirmation countries",
                fontsize=9.2, color=MUTED)

    canvas.text(0.055, 0.810, "A", fontsize=10, color="white", weight="bold", ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.28", "facecolor": INK, "edgecolor": INK})
    canvas.text(0.082, 0.810, "Hypothesised multi-step relation", fontsize=12.3, color=INK, weight="bold", va="center")
    boxes = [
        (0.055, "CITY", "Benghazi", BLUE_LIGHT, BLUE),
        (0.220, "CONTAINING COUNTRY", "Libya", GOLD_LIGHT, GOLD),
        (0.385, "CAPITAL", "Tripoli", TEAL_LIGHT, TEAL),
    ]
    for x, heading, value, face, edge in boxes:
        rounded_box(canvas, x, 0.665, 0.130, 0.095, face=face, edge=edge)
        canvas.text(x + 0.065, 0.728, heading, fontsize=6.8, color=edge, weight="bold", ha="center")
        canvas.text(x + 0.065, 0.690, value, fontsize=11, color=INK, weight="bold", ha="center")
    arrow(canvas, (0.185, 0.712), (0.215, 0.712), color=MUTED)
    arrow(canvas, (0.350, 0.712), (0.380, 0.712), color=MUTED)

    rounded_box(canvas, 0.135, 0.485, 0.300, 0.105, face=SOFT, edge=PURPLE)
    canvas.text(0.285, 0.557, "FROZEN CAPITAL-RELATION PANEL", fontsize=8, color=PURPLE, weight="bold", ha="center")
    canvas.text(0.285, 0.519, "L28F3431   L28F1278   L28F2918", fontsize=9, color=INK,
                family="monospace", weight="bold", ha="center")
    canvas.text(0.285, 0.493, "inhibit the same coordinates in both prompts", fontsize=7.1, color=MUTED, ha="center")
    arrow(canvas, (0.435, 0.538), (0.472, 0.650), color=RED, linewidth=1.6)

    rounded_box(canvas, 0.055, 0.315, 0.215, 0.095, face=RED_LIGHT, edge=RED)
    canvas.text(0.1625, 0.381, "CAPITAL PROMPT", fontsize=7.1, color=RED, weight="bold", ha="center")
    canvas.text(0.1625, 0.346, "logit(Tripoli) - logit(Libya)", fontsize=8.5, color=INK, weight="bold", ha="center")
    rounded_box(canvas, 0.300, 0.315, 0.215, 0.095, face=BLUE_LIGHT, edge=BLUE)
    canvas.text(0.4075, 0.381, "INVERSE CONTROL", fontsize=7.1, color=BLUE, weight="bold", ha="center")
    canvas.text(0.4075, 0.346, "country-containing-city prompt", fontsize=8.2, color=INK, weight="bold", ha="center")

    canvas.text(0.580, 0.810, "B", fontsize=10, color="white", weight="bold", ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.28", "facecolor": INK, "edgecolor": INK})
    canvas.text(0.607, 0.810, "Primary Top-3 confirmation", fontsize=12.3, color=INK, weight="bold", va="center")
    forest = figure.add_axes([0.62, 0.52, 0.325, 0.245])
    forest_plot(
        forest,
        [
            ("Capital prompt", summary["mean_capital_prompt_delta"],
             summary["bootstrap_95_ci_mean_capital_prompt_delta"], RED),
            ("Inverse control", summary["mean_inverse_country_prompt_delta"],
             summary["bootstrap_95_ci_mean_inverse_country_prompt_delta"], BLUE),
            ("Relation-specific", summary["mean_relation_specific_difference"],
             summary["bootstrap_95_ci_mean_relation_specific_difference"], PURPLE),
        ],
        xlabel="Change in capital-minus-country logit gap",
        xlim=(-1.85, 0.22),
    )

    canvas.text(0.580, 0.430, "C", fontsize=10, color="white", weight="bold", ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.28", "facecolor": INK, "edgecolor": INK})
    canvas.text(0.607, 0.430, "Compactness and random controls", fontsize=12.3, color=INK, weight="bold", va="center")
    curve_axis = figure.add_axes([0.62, 0.16, 0.325, 0.225])
    names = ["top_1", "top_3_primary", "top_5", "top_10", "top_20", "all_positive_graph"]
    sizes = [1, 3, 5, 10, 20, payload["candidate_feature_count"]]
    values = [panel_by_name(payload, name)["summary"]["mean_relation_specific_difference"] for name in names]
    curve_axis.plot(range(len(sizes)), values, color=PURPLE, linewidth=2.2, marker="o", markersize=6)
    curve_axis.axhline(0, color=INK, linewidth=0.9)
    random_values = primary["random_control_mean_relation_specific_differences"]
    curve_axis.scatter([1] * len(random_values), random_values, marker="x", s=42, color=GOLD,
                       linewidth=1.5, label="five layer-matched random Top-3 panels", zorder=4)
    curve_axis.scatter([1], [summary["mean_relation_specific_difference"]], s=75, facecolor="white",
                       edgecolor=RED, linewidth=2, zorder=5, label="primary Top-3")
    curve_axis.set_xticks(range(len(sizes)), [str(size) for size in sizes])
    curve_axis.set_ylabel("Relation specificity", fontsize=8.2, color=MUTED)
    curve_axis.set_xlabel("Number of graph features inhibited", fontsize=8.2, color=MUTED)
    curve_axis.set_ylim(-3.35, 0.30)
    format_axis(curve_axis, grid_axis="y")
    curve_axis.legend(frameon=False, fontsize=6.8, loc="lower left")

    rounded_box(canvas, 0.055, 0.145, 0.465, 0.085, face=TEAL_LIGHT, edge=TEAL)
    canvas.text(0.073, 0.198,
                "Primary rule passed: relation-specific effect -1.102",
                fontsize=10, color=INK, weight="bold")
    canvas.text(0.073, 0.166,
                "95% CI [-1.449, -0.750] | stronger than all five random panels",
                fontsize=8.5, color=INK)
    canvas.text(0.055, 0.083,
                "15/16 paired effects were negative; 0/16 interventions changed a capital prediction into the country name.",
                fontsize=8.2, color=INK)
    canvas.text(0.055, 0.052,
                "The result supports a capital-relation-associated influence. It does not by itself establish an explicit city-to-country intermediate step.",
                fontsize=8.1, color=PURPLE, weight="bold")
    save_figure(figure, output_dir, "fig_capitals_10k_relation_panel")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-graph", type=Path,
                        default=Path("outputs/math_large_data_test/math_large_10000_topk256_graph_feature_screen.json"))
    parser.add_argument("--math-localisation", type=Path,
                        default=Path("outputs/math_large_data_test/math_large_10000_topk256_balanced_localisation.json"))
    parser.add_argument("--math-replication", type=Path,
                        default=Path("outputs/math_large_data_test/math_large_10000_topk256_top20_replication.json"))
    parser.add_argument("--capitals", type=Path,
                        default=Path("outputs/capitals_large_data_test/capitals_large_10000_topk128_relation_screen.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("presentation/figures"))
    args = parser.parse_args()

    math_graph = load_json(args.math_graph)
    plot_math_graph_screen(math_graph, args.output_dir)
    plot_math_graph_screen(
        math_graph,
        args.output_dir,
        show_all_random_controls=False,
        stem="fig_math_10k_graph_carry_panel_four_weaker_controls",
    )
    plot_math_balanced(load_json(args.math_localisation), load_json(args.math_replication), args.output_dir)
    plot_capitals_relation(load_json(args.capitals), args.output_dir)
    print(f"Saved follow-up figures under {args.output_dir}")


if __name__ == "__main__":
    main()

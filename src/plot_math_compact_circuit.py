"""Render an evidence-faithful summary of the arithmetic carry feature test.

The left panel shows the frozen Top-10 panel selected from the retained
attribution graph. The right panel shows the disjoint matched-control
confirmation. The result is deliberately presented as a rejected candidate
carry circuit: inhibition affected carry and no-carry arithmetic similarly.
Only completed JSON artifacts are required, so the renderer is CPU-only.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


BACKGROUND = "#F5F7FA"
INK = "#17212B"
MUTED = "#5F6B76"
GRID = "#D8DEE6"
PURPLE = "#7B5AA6"
PURPLE_LIGHT = "#F0EAF6"
TARGET = "#C53B4C"
TARGET_LIGHT = "#F9E7EA"
CONTROL = "#2F6FA3"
CONTROL_LIGHT = "#E7F0F8"
TEAL = "#17847A"
TEAL_LIGHT = "#E4F3F0"
AMBER = "#B7791F"
AMBER_LIGHT = "#FBF1D8"

LAYER_COLOURS = {
    4: "#4C78A8",
    12: "#4E9E98",
    20: "#A27B35",
    24: "#8D6CAB",
    28: "#C45A72",
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def feature_node_id(feature: Dict[str, Any]) -> str:
    return f"layer_{int(feature['layer'])}_feature_{int(feature['feature'])}"


def primary_panel(screen: Dict[str, Any]) -> Dict[str, Any]:
    primary_name = screen["confirmation"]["primary_result"]["panel"]
    return next(
        panel
        for panel in screen["confirmation"]["panels"]
        if panel["name"] == primary_name
    )


def add_round_box(
    axis,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str,
    edge: str = GRID,
    linewidth: float = 1.0,
    radius: float = 0.012,
    linestyle: str = "solid",
    zorder: int = 1,
):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def add_arrow(
    axis,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    colour: str,
    linewidth: float = 1.4,
    alpha: float = 1.0,
    linestyle: str = "solid",
    curve: float = 0.0,
    mutation_scale: float = 10,
    zorder: int = 2,
):
    from matplotlib.patches import FancyArrowPatch

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=colour,
        alpha=alpha,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={curve}",
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    axis.add_patch(arrow)
    return arrow


def draw_feature_node(
    axis,
    x: float,
    y: float,
    *,
    layer: int,
    feature: int,
    rank: int,
    width: float = 0.059,
    height: float = 0.052,
) -> Tuple[float, float, float, float]:
    colour = LAYER_COLOURS[layer]
    add_round_box(
        axis,
        x,
        y,
        width,
        height,
        face="white",
        edge=colour,
        linewidth=1.25,
        radius=0.008,
        zorder=5,
    )
    axis.text(
        x + 0.006,
        y + height * 0.66,
        f"#{rank}",
        ha="left",
        va="center",
        fontsize=6.2,
        color=MUTED,
        zorder=6,
    )
    axis.text(
        x + width * 0.58,
        y + height * 0.44,
        f"F{feature}",
        ha="center",
        va="center",
        fontsize=7.4,
        color=INK,
        weight="bold",
        zorder=6,
    )
    return (x, y, width, height)


def effect_x(value: float, x0: float, x1: float) -> float:
    lower, upper = -0.20, 0.05
    return x0 + (value - lower) / (upper - lower) * (x1 - x0)


def render(graph: Dict[str, Any], screen: Dict[str, Any], output_paths: Sequence[Path]) -> None:
    if screen.get("status") != "complete":
        raise ValueError(f"Feature screen is incomplete: {screen.get('status')!r}")

    panel = primary_panel(screen)
    features = list(panel["features"])
    summary = panel["summary"]
    if len(features) != 10:
        raise ValueError(f"Expected the frozen Top-10 panel, found {len(features)} features")

    graph_nodes = {node["id"]: node for node in graph["nodes"]}
    rank = {feature_node_id(feature): index for index, feature in enumerate(features, start=1)}
    selected_ids = set(rank)
    missing = selected_ids.difference(graph_nodes)
    if missing:
        raise ValueError(f"Frozen features absent from graph: {sorted(missing)}")

    selected_edges = [
        edge
        for edge in graph["edges"]
        if edge["source"] in selected_ids
        and (edge["target"] in selected_ids or edge["target"] == "target_logit")
    ]
    target_edges = [edge for edge in selected_edges if edge["target"] == "target_logit"]
    internal_edges = [edge for edge in selected_edges if edge["target"] in selected_ids]

    target_delta = float(summary["mean_target_delta"])
    control_delta = float(summary["mean_no_carry_control_delta"])
    paired_delta = float(summary["mean_paired_difference"])
    target_ci = [float(value) for value in summary["bootstrap_95_ci_mean_target_delta"]]
    control_ci = [
        float(value)
        for value in summary["bootstrap_95_ci_mean_no_carry_control_delta"]
    ]
    paired_ci = [
        float(value)
        for value in summary["bootstrap_95_ci_mean_paired_difference"]
    ]
    n_cases = int(summary["eligible_cases"])
    n_target_negative = round(float(summary["fraction_target_delta_negative"]) * n_cases)
    n_target_more_negative = round(
        float(summary["fraction_target_more_negative_than_control"]) * n_cases
    )
    n_layers = len({int(feature["layer"]) for feature in features})
    candidate_count = int(screen["candidate_feature_count"])
    discovery_count = len(screen["case_selection"]["discovery_case_keys"])
    supports_specificity = bool(
        screen["confirmation"]["primary_result"][
            "supports_carry_selectivity_under_predeclared_rule"
        ]
    )
    if supports_specificity:
        raise ValueError("This renderer is intended for the recorded failed specificity test")

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

    figure, axis = plt.subplots(figsize=(15.2, 8.4), facecolor=BACKGROUND)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.04,
        0.952,
        "Arithmetic feature inhibition affects digit state, not carry specifically",
        fontsize=18.2,
        weight="bold",
        color=INK,
        va="top",
    )
    axis.text(
        0.04,
        0.912,
        "Attribution-guided TopK SAE panel and frozen matched-control confirmation",
        fontsize=10.5,
        color=MUTED,
        va="top",
    )

    left = (0.035, 0.12, 0.61, 0.75)
    right = (0.67, 0.12, 0.295, 0.75)
    add_round_box(axis, *left, face="white", edge="#E2E6EC", radius=0.018)
    add_round_box(axis, *right, face="white", edge="#E2E6EC", radius=0.018)

    axis.text(
        0.055,
        0.835,
        "A",
        fontsize=10,
        weight="bold",
        color="white",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": INK, "edgecolor": INK},
    )
    axis.text(
        0.078,
        0.835,
        "Candidate panel from one carry graph",
        fontsize=12.5,
        weight="bold",
        color=INK,
        va="center",
    )
    axis.text(
        0.692,
        0.835,
        "B",
        fontsize=10,
        weight="bold",
        color="white",
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": INK, "edgecolor": INK},
    )
    axis.text(
        0.715,
        0.835,
        "Frozen causal specificity test",
        fontsize=12.5,
        weight="bold",
        color=INK,
        va="center",
    )

    # Prompt and graph objective.
    add_round_box(axis, 0.055, 0.548, 0.142, 0.185, face="#F8FAFC", edge=GRID, radius=0.014)
    axis.text(0.068, 0.710, "GRAPH PROMPT", fontsize=7.2, weight="bold", color=MUTED, va="top")
    axis.text(0.068, 0.681, "58 + 83 =", fontsize=11.0, color=INK, weight="bold", va="center")
    add_round_box(axis, 0.139, 0.655, 0.028, 0.038, face=TARGET_LIGHT, edge=TARGET, radius=0.007)
    axis.text(0.153, 0.674, "1", fontsize=11.0, color=TARGET, weight="bold", ha="center", va="center")
    axis.text(0.169, 0.674, "_", fontsize=11.0, color=INK, weight="bold", va="center")
    axis.text(0.068, 0.632, "teacher-forced prefix", fontsize=6.8, color=MUTED, va="center")
    axis.text(0.068, 0.600, "OBJECTIVE", fontsize=6.8, weight="bold", color=MUTED, va="center")
    axis.text(0.068, 0.578, "logit(4) - logit(3)", fontsize=7.5, color=INK, weight="bold", va="center")
    axis.text(0.068, 0.558, "clean value: 7.125", fontsize=7.1, color=MUTED, va="center")

    # Frozen feature panel.
    panel_x, panel_y, panel_w, panel_h = 0.213, 0.303, 0.352, 0.467
    add_round_box(
        axis,
        panel_x,
        panel_y,
        panel_w,
        panel_h,
        face="#FBFAFC",
        edge="#B9A4BF",
        linewidth=1.1,
        radius=0.016,
        linestyle=(0, (4, 3)),
    )
    axis.text(0.230, 0.746, "FROZEN INHIBITION PANEL", fontsize=7.2, weight="bold", color=PURPLE)
    axis.text(0.230, 0.721, f"10 of {candidate_count} positive graph features", fontsize=8.3, color=INK)

    by_layer: Dict[int, List[Dict[str, Any]]] = {}
    for feature in features:
        by_layer.setdefault(int(feature["layer"]), []).append(feature)

    layer_x = {4: 0.224, 12: 0.291, 20: 0.365, 24: 0.431, 28: 0.498}
    layer_y = {
        4: [0.579, 0.487],
        12: [0.620, 0.529, 0.438],
        20: [0.529],
        24: [0.579, 0.487],
        28: [0.579, 0.487],
    }
    for layer, x in layer_x.items():
        axis.text(x + 0.0295, 0.680, f"Layer {layer}", fontsize=6.7, color=MUTED, ha="center")

    node_boxes: Dict[str, Tuple[float, float, float, float]] = {}
    for layer in [4, 12, 20, 24, 28]:
        for feature, y in zip(by_layer[layer], layer_y[layer]):
            node_id = feature_node_id(feature)
            node_boxes[node_id] = draw_feature_node(
                axis,
                layer_x[layer],
                y,
                layer=layer,
                feature=int(feature["feature"]),
                rank=rank[node_id],
            )

    # Compressed input paths and exact retained edges among selected features.
    for feature in by_layer[4]:
        box = node_boxes[feature_node_id(feature)]
        add_arrow(
            axis,
            (0.197, 0.641),
            (box[0], box[1] + box[3] / 2),
            colour="#8A949E",
            linewidth=0.9,
            alpha=0.75,
            linestyle=(0, (3, 3)),
            mutation_scale=7,
            curve=0.08 if box[1] > 0.53 else -0.08,
        )
    max_internal = max(abs(float(edge["weight"])) for edge in internal_edges)
    for edge_index, edge in enumerate(internal_edges):
        source = node_boxes[edge["source"]]
        target = node_boxes[edge["target"]]
        weight = float(edge["weight"])
        relative = abs(weight) / max_internal
        add_arrow(
            axis,
            (source[0] + source[2], source[1] + source[3] / 2),
            (target[0], target[1] + target[3] / 2),
            colour=TEAL if weight >= 0 else TARGET,
            linewidth=0.55 + 1.45 * relative,
            alpha=0.35 + 0.55 * relative,
            mutation_scale=7,
            curve=((edge_index % 3) - 1) * 0.055,
            zorder=3,
        )

    objective_x, objective_y, objective_w, objective_h = 0.579, 0.487, 0.052, 0.157
    add_round_box(
        axis,
        objective_x,
        objective_y,
        objective_w,
        objective_h,
        face=AMBER_LIGHT,
        edge=AMBER,
        linewidth=1.25,
        radius=0.012,
        zorder=5,
    )
    axis.text(objective_x + objective_w / 2, objective_y + objective_h * 0.72, "LOGIT", fontsize=6.8, color=AMBER, weight="bold", ha="center", va="center", zorder=6)
    axis.text(objective_x + objective_w / 2, objective_y + objective_h * 0.47, "4", fontsize=9.3, color=INK, weight="bold", ha="center", va="center", zorder=6)
    axis.text(objective_x + objective_w / 2, objective_y + objective_h * 0.25, "minus 3", fontsize=6.8, color=INK, ha="center", va="center", zorder=6)

    max_target = max(abs(float(edge["weight"])) for edge in target_edges)
    for edge in target_edges:
        source = node_boxes[edge["source"]]
        relative = abs(float(edge["weight"])) / max_target
        add_arrow(
            axis,
            (source[0] + source[2], source[1] + source[3] / 2),
            (objective_x, objective_y + objective_h / 2),
            colour=PURPLE,
            linewidth=0.8 + 1.8 * relative,
            alpha=0.45 + 0.45 * relative,
            mutation_scale=7,
            curve=0.08 if source[1] > 0.53 else -0.08,
            zorder=4,
        )

    # Discovery facts and edge legend.
    stat_y = 0.199
    stat_width = 0.164
    facts = [
        ("CANDIDATES", f"10 / {candidate_count} features", PURPLE, PURPLE_LIGHT),
        ("LAYER SPREAD", f"{n_layers} / 7 SAE layers", CONTROL, CONTROL_LIGHT),
        ("SELECTION DATA", f"{discovery_count} matched pairs", TEAL, TEAL_LIGHT),
    ]
    for index, (heading, value, colour, face) in enumerate(facts):
        x = 0.055 + index * (stat_width + 0.014)
        add_round_box(axis, x, stat_y, stat_width, 0.078, face=face, edge=colour, radius=0.011)
        axis.text(x + 0.012, stat_y + 0.055, heading, fontsize=6.4, weight="bold", color=colour, va="center")
        axis.text(x + 0.012, stat_y + 0.027, value, fontsize=8.1, weight="bold", color=INK, va="center")

    axis.plot([0.058, 0.081], [0.158, 0.158], color=TEAL, linewidth=1.5)
    axis.text(0.088, 0.158, "positive retained edge", fontsize=6.8, color=MUTED, va="center")
    axis.plot([0.211, 0.234], [0.158, 0.158], color=TARGET, linewidth=1.5)
    axis.text(0.241, 0.158, "negative retained edge", fontsize=6.8, color=MUTED, va="center")
    axis.plot([0.378, 0.401], [0.158, 0.158], color=PURPLE, linewidth=1.7)
    axis.text(0.408, 0.158, "feature-to-objective edge", fontsize=6.8, color=MUTED, va="center")

    # Matched confirmation protocol.
    axis.text(0.692, 0.790, "INTERVENTION", fontsize=7.2, weight="bold", color=MUTED)
    add_round_box(axis, 0.692, 0.682, 0.250, 0.078, face=PURPLE_LIGHT, edge=PURPLE, radius=0.011)
    axis.text(0.706, 0.736, "INHIBIT THE SAME 10 LATENTS", fontsize=6.8, color=PURPLE, weight="bold", va="center")
    axis.text(0.706, 0.707, "z_j <- 0 at the final token; reconstruction error preserved", fontsize=7.7, color=INK, va="center")

    add_arrow(axis, (0.817, 0.682), (0.759, 0.625), colour=TARGET, linewidth=1.4)
    add_arrow(axis, (0.817, 0.682), (0.875, 0.625), colour=CONTROL, linewidth=1.4)
    add_round_box(axis, 0.692, 0.549, 0.112, 0.076, face=TARGET_LIGHT, edge=TARGET, radius=0.011)
    axis.text(0.706, 0.602, "CARRY TARGETS", fontsize=6.8, color=TARGET, weight="bold", va="center")
    axis.text(0.706, 0.573, "24 held-out cases", fontsize=8.0, color=INK, va="center")
    add_round_box(axis, 0.830, 0.549, 0.112, 0.076, face=CONTROL_LIGHT, edge=CONTROL, radius=0.011)
    axis.text(0.844, 0.602, "NO-CARRY CONTROLS", fontsize=6.6, color=CONTROL, weight="bold", va="center")
    axis.text(0.844, 0.573, "24 matched controls", fontsize=7.8, color=INK, va="center")

    axis.text(0.692, 0.516, "MEAN CHANGE IN CORRECT-MINUS-CONTRAST GAP", fontsize=6.6, weight="bold", color=MUTED)
    ruler_x0, ruler_x1 = 0.775, 0.929
    track_data = [
        (0.476, target_delta, target_ci, TARGET, f"carry  {target_delta:+.3f}"),
        (0.435, control_delta, control_ci, CONTROL, f"control  {control_delta:+.3f}"),
    ]
    for y, _, _, _, _ in track_data:
        axis.plot([ruler_x0, ruler_x1], [y, y], color=GRID, linewidth=1.1, zorder=1)
    zero_x = effect_x(0.0, ruler_x0, ruler_x1)
    axis.plot([zero_x, zero_x], [0.420, 0.492], color=INK, linewidth=0.9)
    axis.text(zero_x, 0.498, "0", fontsize=6.4, color=MUTED, ha="center")
    for y, value, ci, colour, label in track_data:
        lo_x = effect_x(ci[0], ruler_x0, ruler_x1)
        hi_x = effect_x(ci[1], ruler_x0, ruler_x1)
        point_x = effect_x(value, ruler_x0, ruler_x1)
        axis.text(0.697, y, label, fontsize=7.0, color=colour, weight="bold", ha="left", va="center")
        axis.plot([lo_x, hi_x], [y, y], color=colour, linewidth=2.0, zorder=3)
        axis.plot([lo_x, lo_x], [y - 0.007, y + 0.007], color=colour, linewidth=1.0, zorder=3)
        axis.plot([hi_x, hi_x], [y - 0.007, y + 0.007], color=colour, linewidth=1.0, zorder=3)
        axis.scatter([point_x], [y], s=42, color=colour, edgecolor="white", linewidth=0.65, zorder=4)

    add_round_box(axis, 0.692, 0.235, 0.250, 0.153, face="#FAFBFC", edge=GRID, radius=0.012)
    axis.text(0.706, 0.364, "FROZEN TOP-10 CONFIRMATION", fontsize=7.0, color=INK, weight="bold")
    axis.text(0.706, 0.334, "Paired specificity", fontsize=7.6, color=INK, weight="bold")
    axis.text(0.929, 0.334, f"{paired_delta:+.3f}", fontsize=9.2, color=INK, weight="bold", ha="right")
    axis.text(0.706, 0.306, "Bootstrap 95% CI", fontsize=7.3, color=MUTED)
    axis.text(0.929, 0.306, f"[{paired_ci[0]:+.3f}, {paired_ci[1]:+.3f}]", fontsize=7.7, color=MUTED, weight="bold", ha="right")
    axis.plot([0.706, 0.929], [0.292, 0.292], color=GRID, linewidth=0.8)
    axis.text(0.706, 0.270, f"{n_target_negative}/{n_cases} carry effects negative", fontsize=7.0, color=TARGET)
    axis.text(0.929, 0.270, f"{n_target_more_negative}/{n_cases} beat control", fontsize=7.0, color=CONTROL, ha="right")
    axis.text(0.706, 0.246, "Token flips", fontsize=7.0, color=MUTED)
    axis.text(0.929, 0.246, "0 carry  |  0 control", fontsize=7.2, color=INK, weight="bold", ha="right")

    add_round_box(axis, 0.692, 0.155, 0.250, 0.052, face=AMBER_LIGHT, edge=AMBER, radius=0.011)
    axis.text(0.817, 0.181, "PREDECLARED CARRY-SPECIFICITY RULE: NOT MET", fontsize=7.7, color=AMBER, weight="bold", ha="center", va="center")

    axis.text(
        0.04,
        0.071,
        "Interpretation: inhibition produced a reproducible directional arithmetic effect, but the matched no-carry effect was nearly as large. ",
        fontsize=8.0,
        color=INK,
        va="center",
    )
    axis.text(
        0.04,
        0.040,
        "The panel is therefore carry-associated at discovery, not an identified abstract carry variable. Solid arrows are retained graph edges; dotted input paths compress displayed token edges.",
        fontsize=7.2,
        color=MUTED,
        va="center",
    )

    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".png":
            figure.savefig(output_path, dpi=260, bbox_inches="tight", facecolor=figure.get_facecolor())
        else:
            figure.savefig(output_path, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the compact arithmetic carry feature test")
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("outputs/topk_math_retrain/math_topk256_carry_58_83_4v3_graph.json"),
    )
    parser.add_argument(
        "--screen",
        type=Path,
        default=Path("outputs/topk_math_followup/math_topk256_carry_feature_screen.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/topk_math_followup/figures"),
    )
    parser.add_argument("--stem", default="fig_math_compact_carry_test")
    parser.add_argument(
        "--report-copy",
        type=Path,
        default=Path("report/figures/fig_math_compact_carry_test.pdf"),
        help="Report-ready PDF copy",
    )
    parser.add_argument("--no-report-copy", action="store_true")
    args = parser.parse_args()

    output_paths = [
        args.output_dir / f"{args.stem}.png",
        args.output_dir / f"{args.stem}.pdf",
        args.output_dir / f"{args.stem}.svg",
    ]
    if not args.no_report_copy:
        output_paths.append(args.report_copy)

    render(load_json(args.graph), load_json(args.screen), output_paths)
    for path in output_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

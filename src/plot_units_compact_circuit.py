"""Render an evidence-faithful circuit summary for the units TopK experiment.

The figure combines two distinct objects without conflating them: retained
attribution-graph edges used for feature discovery, and the frozen-panel swap
used for causal confirmation. It requires only completed JSON artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


BACKGROUND = "#FFFFFF"
INK = "#17212B"
MUTED = "#5F6B76"
GRID = "#D8DEE6"
FORCE = "#C53B4C"
FORCE_LIGHT = "#F9E7EA"
MASS = "#2F6FA3"
MASS_LIGHT = "#E7F0F8"
ENERGY = "#17847A"
ENERGY_LIGHT = "#E4F3F0"
PURPLE = "#8D5A9A"
PURPLE_LIGHT = "#F0E7F2"
GOLD = "#C58A13"
GOLD_LIGHT = "#FBF1D8"

LAYER_COLOURS = {
    4: "#4C78A8",
    8: "#4E9E98",
    28: "#9A6B9C",
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def feature_node_id(feature: Dict[str, Any]) -> str:
    return f"layer_{int(feature['layer'])}_feature_{int(feature['feature'])}"


def force_related(outputs: Iterable[str]) -> bool:
    return any("force" in str(value).lower() for value in outputs)


def primary_panel(screen: Dict[str, Any]) -> Dict[str, Any]:
    primary_name = screen["confirmation"]["primary_result"]["panel"]
    return next(
        panel
        for panel in screen["confirmation"]["panels"]
        if panel["name"] == primary_name
    )


def panel_by_name(screen: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next(panel for panel in screen["confirmation"]["panels"] if panel["name"] == name)


def broad_control(screen: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next(
        row for row in screen["confirmation"]["broad_controls"] if row["name"] == name
    )


def confirmation_clean_gaps(screen: Dict[str, Any]) -> List[float]:
    contexts = set(screen["case_selection"]["confirmation_contexts"])
    return [
        float(row["target_clean"]["gap"])
        for row in screen["case_selection"]["baseline_screened_cases"]
        if row["context"] in contexts
    ]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


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
    semantic: bool,
    width: float = 0.074,
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
        radius=0.009,
        zorder=4,
    )
    axis.text(
        x + 0.008,
        y + height * 0.72,
        f"#{rank}",
        ha="left",
        va="center",
        fontsize=6.4,
        color=MUTED,
        zorder=5,
    )
    axis.text(
        x + width * 0.61,
        y + height * 0.40,
        f"F{feature}",
        ha="center",
        va="center",
        fontsize=8.2,
        color=INK,
        weight="bold",
        zorder=5,
    )
    if semantic:
        axis.scatter(
            [x + width - 0.009],
            [y + height - 0.009],
            s=18,
            color=FORCE,
            edgecolor="white",
            linewidth=0.45,
            zorder=6,
        )
    return (x, y, width, height)


def render(graph: Dict[str, Any], screen: Dict[str, Any], output_paths: Sequence[Path]) -> None:
    if screen.get("status") != "complete":
        raise ValueError(f"Feature screen is incomplete: {screen.get('status')!r}")

    primary = primary_panel(screen)
    summary = primary["summary"]
    features = list(primary["features"])
    if len(features) != 10:
        raise ValueError(f"Expected the frozen Top-10 panel, found {len(features)} features")

    graph_nodes = {node["id"]: node for node in graph["nodes"]}
    rank = {feature_node_id(feature): index for index, feature in enumerate(features, start=1)}
    selected_ids = set(rank)
    selected_nodes = {node_id: graph_nodes[node_id] for node_id in selected_ids}
    target_edges = {
        edge["source"]: float(edge["weight"])
        for edge in graph["edges"]
        if edge["target"] == "target_logit" and edge["source"] in selected_ids
    }

    clean_gap = mean(confirmation_clean_gaps(screen))
    force_gap = clean_gap + float(summary["mean_force_source_delta"])
    mass_gap = clean_gap + float(summary["mean_mass_source_delta"])
    all_graph_effect = float(
        panel_by_name(screen, "all_positive_graph")["summary"]["mean_force_source_delta"]
    )
    full_latent_effect = float(
        broad_control(screen, "full_latent_swap")["summary"]["mean_force_source_delta"]
    )
    graph_effect_ratio = float(summary["mean_force_source_delta"]) / all_graph_effect
    latent_effect_ratio = float(summary["mean_force_source_delta"]) / full_latent_effect
    semantic_count = sum(
        force_related(selected_nodes[feature_node_id(feature)].get("top_outputs", []))
        for feature in features
    )

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
        "Compact force-to-unit transfer in Qwen3-4B-Instruct",
        fontsize=19,
        weight="bold",
        color=INK,
        va="top",
    )
    axis.text(
        0.04,
        0.912,
        "Attribution-guided TopK SAE panel and frozen swap-in confirmation",
        fontsize=10.5,
        color=MUTED,
        va="top",
    )

    left = (0.035, 0.12, 0.61, 0.75)
    right = (0.67, 0.12, 0.295, 0.75)
    add_round_box(axis, *left, face="white", edge="#E2E6EC", linewidth=1.0, radius=0.018)
    add_round_box(axis, *right, face="white", edge="#E2E6EC", linewidth=1.0, radius=0.018)

    axis.text(0.055, 0.835, "A", fontsize=10, weight="bold", color="white", ha="center", va="center",
              bbox={"boxstyle": "round,pad=0.28", "facecolor": INK, "edgecolor": INK})
    axis.text(0.078, 0.835, "Attribution-guided feature panel", fontsize=12.5, weight="bold",
              color=INK, va="center")
    axis.text(0.692, 0.835, "B", fontsize=10, weight="bold", color="white", ha="center", va="center",
              bbox={"boxstyle": "round,pad=0.28", "facecolor": INK, "edgecolor": INK})
    axis.text(0.715, 0.835, "Frozen causal confirmation", fontsize=12.5, weight="bold",
              color=INK, va="center")

    # Graph prompt and objective.
    add_round_box(axis, 0.055, 0.545, 0.150, 0.18, face="#F8FAFC", edge=GRID, radius=0.014)
    axis.text(0.068, 0.702, "GRAPH PROMPT", fontsize=7.2, weight="bold", color=MUTED, va="top")
    axis.text(0.068, 0.682, "Official SI unit for", fontsize=8.4, color=INK, va="top")
    add_round_box(axis, 0.067, 0.610, 0.055, 0.035, face=FORCE_LIGHT, edge=FORCE, radius=0.008)
    axis.text(0.0945, 0.6275, "force", fontsize=9.2, weight="bold", color=FORCE,
              ha="center", va="center")
    axis.text(0.068, 0.588, "of a moving engine\nthrust is named \"", fontsize=8.1,
              color=INK, va="top", linespacing=1.25)

    # Frozen panel enclosure.
    add_round_box(
        axis,
        0.220,
        0.315,
        0.340,
        0.445,
        face="#FBFAFC",
        edge="#B9A4BF",
        linewidth=1.1,
        radius=0.016,
        linestyle=(0, (4, 3)),
        zorder=1,
    )
    axis.text(0.238, 0.736, "FROZEN SWAP PANEL", fontsize=7.2, weight="bold", color=PURPLE)
    axis.text(0.238, 0.712, "10 of 73 positive graph features", fontsize=8.3, color=INK)

    axis.text(0.269, 0.675, "Layer 4", fontsize=7.2, color=MUTED, ha="center")
    axis.text(0.357, 0.675, "Layer 8", fontsize=7.2, color=MUTED, ha="center")
    axis.text(0.475, 0.675, "Layer 28", fontsize=7.2, color=MUTED, ha="center")

    by_layer: Dict[int, List[Dict[str, Any]]] = {}
    for feature in features:
        by_layer.setdefault(int(feature["layer"]), []).append(feature)

    node_boxes: Dict[str, Tuple[float, float, float, float]] = {}
    for layer, x, y in [(4, 0.232, 0.565), (8, 0.320, 0.565)]:
        feature = by_layer[layer][0]
        node_id = feature_node_id(feature)
        node_boxes[node_id] = draw_feature_node(
            axis,
            x,
            y,
            layer=layer,
            feature=int(feature["feature"]),
            rank=rank[node_id],
            semantic=force_related(selected_nodes[node_id].get("top_outputs", [])),
        )

    layer_28 = by_layer[28]
    y_values = [0.604, 0.536, 0.468, 0.400]
    for index, feature in enumerate(layer_28):
        x = 0.397 if index < 4 else 0.478
        y = y_values[index % 4]
        node_id = feature_node_id(feature)
        node_boxes[node_id] = draw_feature_node(
            axis,
            x,
            y,
            layer=28,
            feature=int(feature["feature"]),
            rank=rank[node_id],
            semantic=force_related(selected_nodes[node_id].get("top_outputs", [])),
        )

    # A compressed but explicit path from the highlighted input to the early features.
    add_arrow(axis, (0.205, 0.635), (0.232, 0.591), colour="#8A949E", linewidth=1.1)
    l4_box = node_boxes[feature_node_id(by_layer[4][0])]
    l8_box = node_boxes[feature_node_id(by_layer[8][0])]
    add_arrow(
        axis,
        (l4_box[0] + l4_box[2], l4_box[1] + l4_box[3] / 2),
        (l8_box[0], l8_box[1] + l8_box[3] / 2),
        colour=LAYER_COLOURS[8],
        linewidth=1.0,
    )
    add_arrow(
        axis,
        (l8_box[0] + l8_box[2], l8_box[1] + l8_box[3] / 2),
        (0.397, 0.591),
        colour="#8A949E",
        linewidth=1.0,
        linestyle=(0, (3, 3)),
    )

    # Retained graph edges from selected layer-28 features to the objective.
    objective_x, objective_y, objective_w, objective_h = 0.574, 0.490, 0.058, 0.155
    add_round_box(
        axis,
        objective_x,
        objective_y,
        objective_w,
        objective_h,
        face=GOLD_LIGHT,
        edge=GOLD,
        linewidth=1.25,
        radius=0.012,
        zorder=4,
    )
    axis.text(objective_x + objective_w / 2, objective_y + objective_h * 0.72, "LOGIT",
              fontsize=7, color=GOLD, weight="bold", ha="center", va="center", zorder=5)
    axis.text(objective_x + objective_w / 2, objective_y + objective_h * 0.44,
              "newtons", fontsize=7.3, color=INK, weight="bold", ha="center", va="center", zorder=5)
    axis.text(objective_x + objective_w / 2, objective_y + objective_h * 0.25,
              "minus\njoules", fontsize=7.1, color=INK, ha="center", va="center", zorder=5,
              linespacing=1.1)

    max_edge = max(abs(value) for value in target_edges.values())
    for node_id, edge_weight in target_edges.items():
        x, y, width, height = node_boxes[node_id]
        relative = abs(edge_weight) / max_edge
        add_arrow(
            axis,
            (x + width, y + height / 2),
            (objective_x, objective_y + objective_h / 2),
            colour=PURPLE,
            linewidth=0.6 + 2.2 * relative,
            alpha=0.35 + 0.55 * relative,
            mutation_scale=7,
            curve=(y + height / 2 - (objective_y + objective_h / 2)) * 0.18,
            zorder=3,
        )

    # Compactness and semantic evidence strip.
    stat_y = 0.205
    stat_width = 0.164
    for index, (heading, value, colour, face) in enumerate(
        [
            ("PANEL", "8/10 at layer 28", PURPLE, PURPLE_LIGHT),
            ("DECODER EVIDENCE", f"{semantic_count}/10 force-related", FORCE, FORCE_LIGHT),
            ("COMPACT EFFECT", f"{100 * graph_effect_ratio:.1f}% of 73-feature shift", ENERGY, ENERGY_LIGHT),
        ]
    ):
        x = 0.055 + index * (stat_width + 0.014)
        add_round_box(axis, x, stat_y, stat_width, 0.078, face=face, edge=colour, radius=0.011)
        axis.text(x + 0.012, stat_y + 0.055, heading, fontsize=6.4, weight="bold", color=colour,
                  va="center")
        axis.text(x + 0.012, stat_y + 0.027, value, fontsize=8.1, weight="bold", color=INK,
                  va="center")

    axis.scatter([0.058], [0.165], s=20, color=FORCE, edgecolor="white", linewidth=0.4)
    axis.text(0.070, 0.165, "force-related decoder output", fontsize=7.2, color=MUTED, va="center")
    axis.plot([0.250, 0.274], [0.165, 0.165], color=PURPLE, linewidth=1.6)
    axis.text(0.281, 0.165, "retained layer-28-to-logit edge", fontsize=7.2, color=MUTED,
              va="center")

    # Confirmation panel: donors, swap operation and unchanged top prediction.
    axis.text(0.692, 0.790, "ACTIVATION DONORS", fontsize=7.2, weight="bold", color=MUTED)
    add_round_box(axis, 0.692, 0.686, 0.112, 0.076, face=FORCE_LIGHT, edge=FORCE, radius=0.011)
    axis.text(0.748, 0.739, "FORCE SOURCE", fontsize=6.8, color=FORCE, weight="bold", ha="center", va="center")
    axis.text(0.748, 0.710, "newtons-predicting", fontsize=7.6, color=INK, ha="center", va="center")
    add_round_box(axis, 0.830, 0.686, 0.112, 0.076, face=MASS_LIGHT, edge=MASS, radius=0.011)
    axis.text(0.886, 0.739, "MASS CONTROL", fontsize=6.8, color=MASS, weight="bold", ha="center", va="center")
    axis.text(0.886, 0.710, "kilograms-predicting", fontsize=7.6, color=INK, ha="center", va="center")

    add_arrow(axis, (0.748, 0.686), (0.794, 0.627), colour=FORCE, linewidth=1.5)
    add_arrow(axis, (0.886, 0.686), (0.840, 0.627), colour=MASS, linewidth=1.5)
    add_round_box(axis, 0.764, 0.542, 0.106, 0.084, face=PURPLE_LIGHT, edge=PURPLE, radius=0.012)
    axis.text(0.817, 0.598, "SWAP-IN", fontsize=7.0, color=PURPLE, weight="bold", ha="center")
    axis.text(0.817, 0.565, "same 10 latents\nat final token\none donor per run", fontsize=6.8, color=INK,
              ha="center", va="center", linespacing=1.05)
    add_arrow(axis, (0.817, 0.542), (0.817, 0.493), colour=ENERGY, linewidth=1.5)
    add_round_box(axis, 0.692, 0.416, 0.250, 0.077, face=ENERGY_LIGHT, edge=ENERGY, radius=0.011)
    axis.text(0.706, 0.469, "ENERGY RECIPIENT", fontsize=6.8, color=ENERGY, weight="bold", va="center")
    axis.text(0.817, 0.440, "16 disjoint systems\nclean answer remains joules", fontsize=7.4,
              color=INK, ha="center", va="center", linespacing=1.12)

    # Mean logit-gap ruler. Separate tracks keep the nearly identical clean and
    # mass-control values legible rather than hiding one marker behind another.
    ruler_x0, ruler_x1 = 0.801, 0.929
    axis.text(0.692, 0.393, "MEAN NEWTONS-MINUS-JOULES GAP", fontsize=6.6, weight="bold", color=MUTED)
    track_values = [
        (0.365, clean_gap, "#7C8792", f"clean  {clean_gap:.2f}"),
        (0.341, force_gap, FORCE, f"force swap  {force_gap:.2f}"),
        (0.317, mass_gap, MASS, f"mass control  {mass_gap:.2f}"),
    ]
    for y, _, _, _ in track_values:
        axis.plot([ruler_x0, ruler_x1], [y, y], color=GRID, linewidth=1.1, zorder=1)
    axis.plot([ruler_x1, ruler_x1], [0.306, 0.377], color=INK, linewidth=1.0)
    axis.text(ruler_x1, 0.383, "0 (boundary)", fontsize=6.4, color=MUTED, ha="right")

    def gap_x(value: float) -> float:
        lower, upper = -12.0, 0.0
        return ruler_x0 + (value - lower) / (upper - lower) * (ruler_x1 - ruler_x0)

    for y, value, colour, label in track_values:
        x = gap_x(value)
        axis.text(0.697, y, label, fontsize=6.8, color=colour, weight="bold",
                  ha="left", va="center")
        axis.scatter([x], [y], s=38, color=colour, edgecolor="white", linewidth=0.65, zorder=3)

    # Primary confirmation result.
    add_round_box(axis, 0.692, 0.178, 0.250, 0.125, face="#FAFBFC", edge=GRID, radius=0.012)
    force_ci = summary["bootstrap_95_ci_mean_force_source_delta"]
    mass_ci = summary["bootstrap_95_ci_mean_mass_source_delta"]
    paired_ci = summary["bootstrap_95_ci_mean_force_minus_mass_difference"]
    axis.text(0.706, 0.279, "FROZEN TOP-10 CONFIRMATION  |  16/16", fontsize=7.0,
              color=INK, weight="bold")
    axis.text(0.706, 0.251, "Force-source shift", fontsize=7.5, color=MUTED)
    axis.text(0.929, 0.251, f"{summary['mean_force_source_delta']:+.3f}", fontsize=8.2,
              color=FORCE, weight="bold", ha="right")
    axis.text(0.706, 0.225, "Mass-control shift", fontsize=7.5, color=MUTED)
    axis.text(0.929, 0.225, f"{summary['mean_mass_source_delta']:+.3f}", fontsize=8.2,
              color=MASS, weight="bold", ha="right")
    axis.plot([0.706, 0.929], [0.212, 0.212], color=GRID, linewidth=0.8)
    axis.text(0.706, 0.192, "Paired specificity", fontsize=7.7, color=INK, weight="bold")
    axis.text(0.929, 0.192, f"{summary['mean_force_minus_mass_difference']:+.3f}", fontsize=9.0,
              color=INK, weight="bold", ha="right")

    axis.text(
        0.817,
        0.151,
        f"95% CIs: force [{force_ci[0]:+.3f}, {force_ci[1]:+.3f}]   |   "
        f"mass [{mass_ci[0]:+.3f}, {mass_ci[1]:+.3f}]",
        fontsize=6.2,
        color=MUTED,
        ha="center",
    )
    axis.text(
        0.817,
        0.132,
        f"paired [{paired_ci[0]:+.3f}, {paired_ci[1]:+.3f}]",
        fontsize=6.2,
        color=MUTED,
        ha="center",
    )

    axis.text(
        0.04,
        0.072,
        "Interpretation: the frozen panel produced a selective logit shift, not a token flip. "
        f"Its force-source effect was {100 * latent_effect_ratio:.1f}% of the full-latent swap effect. "
        "Effect ratios are intervention comparisons, not variance explained.",
        fontsize=8.0,
        color=INK,
        va="center",
    )
    axis.text(
        0.04,
        0.038,
        "Solid layer-28 arrows are retained graph edges. The dotted path compresses unshown graph nodes; "
        "the diagram is an evidence summary rather than a complete computational circuit.",
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
    parser = argparse.ArgumentParser(description="Plot the compact units causal circuit summary")
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("outputs/topk_units_retrain/units_topk128_force_graph.json"),
    )
    parser.add_argument(
        "--screen",
        type=Path,
        default=Path("outputs/topk_units_retrain/units_topk128_feature_screen.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/topk_units_retrain/figures"),
    )
    parser.add_argument("--stem", default="fig_units_compact_causal_circuit")
    parser.add_argument(
        "--report-copy",
        type=Path,
        default=Path("report/figures/fig_units_compact_causal_circuit.pdf"),
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

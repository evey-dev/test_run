"""Render a compact, report-ready view of a saved attribution graph.

The JSON graph remains the complete analysis artifact.  This renderer selects a
small backward-connected subset so that the layered structure can be inspected
on a printed page without implying that omitted nodes were absent.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


Node = Dict[str, Any]
Edge = Dict[str, Any]


def layer_sort_key(layer: str) -> Tuple[int, int]:
    if layer == "input":
        return (0, 0)
    if layer == "logits":
        return (2, 0)
    match = re.fullmatch(r"layer_(\d+)", layer)
    if match:
        return (1, int(match.group(1)))
    return (1, 10_000)


def group_nodes(nodes: Iterable[Node]) -> Dict[str, List[Node]]:
    grouped: Dict[str, List[Node]] = defaultdict(list)
    for node in nodes:
        grouped[str(node["layer"])].append(node)
    return grouped


def backward_connected_subset(
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    nodes_per_layer: int,
) -> Tuple[List[str], Set[str]]:
    """Select high-weight predecessors recursively from the output contrast."""

    grouped = group_nodes(nodes)
    layers = sorted(grouped, key=layer_sort_key)
    node_by_id = {str(node["id"]): node for node in nodes}
    selected: Dict[str, Set[str]] = {layer: set() for layer in layers}

    output_layer = layers[-1]
    output_nodes = sorted(
        grouped[output_layer],
        key=lambda node: abs(float(node.get("attribution", 0.0))),
        reverse=True,
    )
    selected[output_layer].update(str(node["id"]) for node in output_nodes[:nodes_per_layer])

    for previous_layer, current_layer in zip(reversed(layers[:-1]), reversed(layers[1:])):
        scores: Dict[str, float] = defaultdict(float)
        for edge in edges:
            source = str(edge["source"])
            target = str(edge["target"])
            source_node = node_by_id.get(source)
            target_node = node_by_id.get(target)
            if source_node is None or target_node is None:
                continue
            if (
                str(source_node["layer"]) == previous_layer
                and str(target_node["layer"]) == current_layer
                and target in selected[current_layer]
            ):
                scores[source] += abs(float(edge.get("weight", 0.0)))

        ranked = sorted(
            grouped[previous_layer],
            key=lambda node: (
                scores.get(str(node["id"]), 0.0),
                abs(float(node.get("attribution", 0.0))),
            ),
            reverse=True,
        )
        connected = [node for node in ranked if scores.get(str(node["id"]), 0.0) > 0.0]
        chosen = connected[:nodes_per_layer]
        if len(chosen) < nodes_per_layer:
            chosen_ids = {str(node["id"]) for node in chosen}
            chosen.extend(
                node
                for node in ranked
                if str(node["id"]) not in chosen_ids
            )
            chosen = chosen[:nodes_per_layer]
        selected[previous_layer].update(str(node["id"]) for node in chosen)

    selected_ids = set().union(*selected.values())
    return layers, selected_ids


def retained_edges(
    edges: Sequence[Edge],
    selected_ids: Set[str],
    node_by_id: Mapping[str, Node],
    edges_per_transition: int,
) -> List[Edge]:
    grouped: Dict[Tuple[str, str], List[Edge]] = defaultdict(list)
    for edge in edges:
        source = str(edge["source"])
        target = str(edge["target"])
        if source not in selected_ids or target not in selected_ids:
            continue
        pair = (str(node_by_id[source]["layer"]), str(node_by_id[target]["layer"]))
        grouped[pair].append(edge)

    retained: List[Edge] = []
    for transition_edges in grouped.values():
        retained.extend(
            sorted(
                transition_edges,
                key=lambda edge: abs(float(edge.get("weight", 0.0))),
                reverse=True,
            )[:edges_per_transition]
        )
    return retained


def node_label(node: Node, target: str, contrast: str) -> str:
    layer = str(node["layer"])
    if layer == "input":
        label = str(node.get("label", node["id"]))
        return label if len(label) <= 13 else label[:12] + "..."
    if layer == "logits":
        return f"{target} - {contrast}" if contrast else target
    match = re.search(r"feature_(\d+)$", str(node["id"]))
    return f"F{match.group(1)}" if match else str(node.get("label", node["id"]))


def layer_label(layer: str) -> str:
    if layer == "input":
        return "Input token"
    if layer == "logits":
        return "Logit contrast"
    match = re.fullmatch(r"layer_(\d+)", layer)
    return f"Layer {match.group(1)}" if match else layer


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def render_graph(
    payload: Dict[str, Any],
    output_dir: Path,
    stem: str,
    nodes_per_layer: int,
    edges_per_transition: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    nodes: List[Node] = payload["nodes"]
    edges: List[Edge] = payload["edges"]
    node_by_id = {str(node["id"]): node for node in nodes}
    layers, selected_ids = backward_connected_subset(nodes, edges, nodes_per_layer)
    selected_edges = retained_edges(
        edges, selected_ids, node_by_id, edges_per_transition
    )

    selected_by_layer: Dict[str, List[Node]] = defaultdict(list)
    for node_id in selected_ids:
        selected_by_layer[str(node_by_id[node_id]["layer"])].append(node_by_id[node_id])
    for layer in layers:
        selected_by_layer[layer].sort(
            key=lambda node: float(node.get("attribution", 0.0)), reverse=True
        )

    positions: Dict[str, Tuple[float, float]] = {}
    for x_position, layer in enumerate(layers):
        layer_nodes = selected_by_layer[layer]
        count = len(layer_nodes)
        if count == 1:
            y_positions = [0.5]
        else:
            y_positions = [0.88 - index * 0.76 / (count - 1) for index in range(count)]
        for node, y_position in zip(layer_nodes, y_positions):
            positions[str(node["id"])] = (float(x_position), y_position)

    configure_matplotlib()
    # Keep the source canvas close to A4 text width.  A much wider canvas would
    # make labels illegible after LaTeX scales the PDF back to \textwidth.
    figure_width = max(7.2, 0.80 * len(layers))
    fig, axis = plt.subplots(figsize=(figure_width, 4.3))
    axis.set_xlim(-0.55, len(layers) - 0.45)
    axis.set_ylim(-0.08, 1.08)
    axis.axis("off")

    transition_max: Dict[Tuple[str, str], float] = defaultdict(float)
    for edge in selected_edges:
        source_layer = str(node_by_id[str(edge["source"])]["layer"])
        target_layer = str(node_by_id[str(edge["target"])]["layer"])
        pair = (source_layer, target_layer)
        transition_max[pair] = max(transition_max[pair], abs(float(edge["weight"])))

    positive = "#b2182b"
    negative = "#2166ac"
    for edge in selected_edges:
        source = str(edge["source"])
        target = str(edge["target"])
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        weight = float(edge["weight"])
        pair = (
            str(node_by_id[source]["layer"]),
            str(node_by_id[target]["layer"]),
        )
        relative = abs(weight) / max(transition_max[pair], 1e-12)
        axis.plot(
            [x0 + 0.30, x1 - 0.30],
            [y0, y1],
            color=positive if weight >= 0 else negative,
            alpha=0.18 + 0.52 * math.sqrt(relative),
            linewidth=0.45 + 1.9 * math.sqrt(relative),
            solid_capstyle="round",
            zorder=1,
        )

    target = str(payload.get("target", "target"))
    contrast = str(payload.get("contrast_target", ""))
    for layer_index, layer in enumerate(layers):
        axis.text(
            layer_index,
            1.025,
            layer_label(layer),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#263238",
        )
        for node in selected_by_layer[layer]:
            node_id = str(node["id"])
            x_position, y_position = positions[node_id]
            attribution = float(node.get("attribution", 0.0))
            colour = positive if attribution >= 0 else negative
            width = 0.69 if layer not in {"input", "logits"} else 0.80
            box = FancyBboxPatch(
                (x_position - width / 2, y_position - 0.047),
                width,
                0.094,
                boxstyle="round,pad=0.008,rounding_size=0.015",
                linewidth=0.65,
                edgecolor=colour,
                facecolor=colour,
                alpha=0.82,
                zorder=3,
            )
            axis.add_patch(box)
            axis.text(
                x_position,
                y_position,
                node_label(node, target, contrast),
                ha="center",
                va="center",
                fontsize=6.6,
                color="white",
                zorder=4,
            )

    objective = str(payload.get("attribution_objective", "Attribution objective"))
    axis.set_title(
        f"Backward-connected visual subset of the carry attribution graph\n{objective}",
        fontsize=10.5,
        fontweight="bold",
        pad=12,
    )
    axis.legend(
        handles=[
            Line2D([0], [0], color=positive, lw=2, label="Positive attribution"),
            Line2D([0], [0], color=negative, lw=2, label="Negative attribution"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=7.5,
    )
    axis.text(
        0.5,
        -0.075,
        f"Displayed subset: up to {nodes_per_layer} nodes per stage and "
        f"{edges_per_transition} edges per transition; complete graph retained in JSON/HTML.",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color="#555555",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_dir / f'{stem}.pdf'}")
    print(f"Selected {len(selected_ids)} of {len(nodes)} nodes and "
          f"{len(selected_edges)} of {len(edges)} edges")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a layered attribution-graph subset")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/report_figures"))
    parser.add_argument("--stem", default="fig_math_attribution_graph")
    parser.add_argument("--nodes-per-layer", type=int, default=5)
    parser.add_argument("--edges-per-transition", type=int, default=14)
    args = parser.parse_args()

    if args.nodes_per_layer < 1 or args.edges_per_transition < 1:
        parser.error("node and edge limits must both be positive")
    with args.graph.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    render_graph(
        payload,
        args.output_dir,
        args.stem,
        args.nodes_per_layer,
        args.edges_per_transition,
    )


if __name__ == "__main__":
    main()

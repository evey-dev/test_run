"""Render a slide-ready causal summary of the 10k units SAE experiment.

The renderer is CPU-only and reads completed graph, feature-screen, and
comparison JSON artifacts. It depicts measured logit-gap effects and explicitly
does not imply a categorical answer swap.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


INK = "#17212B"
MUTED = "#5F6B76"
GRID = "#D8DEE6"
SOFT = "#F6F8FA"
FORCE = "#C53B4C"
FORCE_LIGHT = "#F9E7EA"
MASS = "#2F6FA3"
MASS_LIGHT = "#E7F0F8"
ENERGY = "#17847A"
ENERGY_LIGHT = "#E4F3F0"
PANEL = "#7B5AA6"
PANEL_LIGHT = "#F0EAF6"
GOLD = "#B7791F"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def confirmation_panel(screen: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next(row for row in screen["confirmation"]["panels"] if row["name"] == name)


def broad_control(screen: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next(
        row for row in screen["confirmation"]["broad_controls"] if row["name"] == name
    )


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def clean_confirmation_gap(screen: Dict[str, Any]) -> float:
    contexts = set(screen["case_selection"]["confirmation_contexts"])
    return mean(
        float(row["target_clean"]["gap"])
        for row in screen["case_selection"]["baseline_screened_cases"]
        if row["context"] in contexts
    )


def feature_node_id(feature: Dict[str, Any]) -> str:
    return f"layer_{int(feature['layer'])}_feature_{int(feature['feature'])}"


def ascii_decoder_label(outputs: Sequence[str]) -> str:
    cleaned = [str(value).strip("'\"") for value in outputs]
    preferred = [value for value in cleaned if value.isascii() and value.lower() == "force"]
    if preferred:
        return preferred[0]
    fallback = [value for value in cleaned if value.isascii() and value]
    return fallback[0] if fallback else "decoder direction"


def round_box(axis, x: float, y: float, width: float, height: float, *, face: str, edge: str):
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.009",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.15,
    )
    axis.add_patch(box)
    return box


def arrow(
    axis,
    start,
    end,
    *,
    colour: str,
    linewidth: float = 1.5,
    linestyle: str = "solid",
    curve: float = 0.0,
):
    from matplotlib.patches import FancyArrowPatch

    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=linewidth,
        linestyle=linestyle,
        color=colour,
        connectionstyle=f"arc3,rad={curve}",
        shrinkA=1,
        shrinkB=1,
    )
    axis.add_patch(patch)
    return patch


def configure_matplotlib() -> None:
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


def effect_series(screen: Dict[str, Any]) -> List[Dict[str, Any]]:
    top3 = confirmation_panel(screen, "top_3")["summary"]
    return [
        {
            "label": "Force donor",
            "value": float(top3["mean_force_source_delta"]),
            "ci": [float(value) for value in top3["bootstrap_95_ci_mean_force_source_delta"]],
            "colour": FORCE,
        },
        {
            "label": "Mass control",
            "value": float(top3["mean_mass_source_delta"]),
            "ci": [float(value) for value in top3["bootstrap_95_ci_mean_mass_source_delta"]],
            "colour": MASS,
        },
        {
            "label": "Paired difference",
            "value": float(top3["mean_force_minus_mass_difference"]),
            "ci": [
                float(value)
                for value in top3["bootstrap_95_ci_mean_force_minus_mass_difference"]
            ],
            "colour": PANEL,
        },
    ]


def compactness_series(screen: Dict[str, Any]) -> tuple[List[str], List[float]]:
    panel_names = ["top_1", "top_3", "top_5", "top_10_primary", "top_20"]
    labels = ["1", "3", "5", "10", "20", "69", "Full"]
    values = [
        float(
            confirmation_panel(screen, name)["summary"]["mean_force_minus_mass_difference"]
        )
        for name in panel_names
    ]
    values.append(
        float(
            confirmation_panel(screen, "all_positive_graph")["summary"][
                "mean_force_minus_mass_difference"
            ]
        )
    )
    values.append(
        float(
            broad_control(screen, "full_latent_swap")["summary"][
                "mean_force_minus_mass_difference"
            ]
        )
    )
    return labels, values


def render(
    graph: Dict[str, Any],
    screen: Dict[str, Any],
    comparison: Dict[str, Any],
    output_paths: Sequence[Path],
) -> None:
    if screen.get("status") != "complete":
        raise ValueError(f"Feature screen is incomplete: {screen.get('status')!r}")

    top3 = confirmation_panel(screen, "top_3")
    top3_summary = top3["summary"]
    features = list(top3["features"])
    if len(features) != 3 or any(int(feature["layer"]) != 28 for feature in features):
        raise ValueError("Expected the confirmed Top-3 panel to contain three layer-28 features")

    graph_nodes = {node["id"]: node for node in graph["nodes"]}
    feature_rows = []
    for rank, feature in enumerate(features, start=1):
        node = graph_nodes[feature_node_id(feature)]
        feature_rows.append(
            {
                "rank": rank,
                "feature": int(feature["feature"]),
                "attribution": float(node["attribution"]),
                "decoder": ascii_decoder_label(node.get("top_outputs", [])),
            }
        )

    comparison_by_name = {row["run"]: row for row in comparison["runs"]}
    old_effect = float(comparison_by_name["units_topk128_1k"]["primary_force_minus_mass"])
    new_effect = float(comparison_by_name["units_topk128_10k"]["primary_force_minus_mass"])
    clean_gap = clean_confirmation_gap(screen)
    force_gap = clean_gap + float(top3_summary["mean_force_source_delta"])
    mass_gap = clean_gap + float(top3_summary["mean_mass_source_delta"])

    configure_matplotlib()
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(15.6, 8.4), facecolor="white")
    canvas = figure.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")

    canvas.text(
        0.045,
        0.945,
        "A three-feature panel transfers a force-associated unit signal",
        fontsize=21,
        color=INK,
        weight="bold",
        va="top",
    )
    canvas.text(
        0.045,
        0.902,
        "10,000-prompt TopK-128 SAEs  |  final-token swap  |  16 held-out systems",
        fontsize=11,
        color=MUTED,
        va="top",
    )
    canvas.plot([0.565, 0.565], [0.13, 0.855], color=GRID, linewidth=1.0)

    canvas.text(0.047, 0.835, "A", fontsize=10, color="white", weight="bold", ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.28", "facecolor": INK, "edgecolor": INK})
    canvas.text(0.073, 0.835, "Causal patch design", fontsize=13, color=INK, weight="bold", va="center")

    round_box(canvas, 0.050, 0.650, 0.145, 0.105, face=FORCE_LIGHT, edge=FORCE)
    canvas.text(0.1225, 0.729, "FORCE DONOR", fontsize=7.2, color=FORCE, weight="bold", ha="center")
    canvas.text(0.1225, 0.694, "force  ->  newtons", fontsize=9.4, color=INK, weight="bold", ha="center")
    canvas.text(0.1225, 0.667, "source activations", fontsize=7.4, color=MUTED, ha="center")

    round_box(canvas, 0.050, 0.447, 0.145, 0.105, face=MASS_LIGHT, edge=MASS)
    canvas.text(0.1225, 0.526, "MATCHED CONTROL", fontsize=7.2, color=MASS, weight="bold", ha="center")
    canvas.text(0.1225, 0.491, "mass  ->  kilograms", fontsize=9.0, color=INK, weight="bold", ha="center")
    canvas.text(0.1225, 0.464, "same target, separate run", fontsize=7.2, color=MUTED, ha="center")

    round_box(canvas, 0.242, 0.435, 0.135, 0.335, face=PANEL_LIGHT, edge=PANEL)
    canvas.text(0.3095, 0.744, "TOP-3 PANEL", fontsize=7.4, color=PANEL, weight="bold", ha="center")
    canvas.text(0.3095, 0.720, "Layer 28", fontsize=8.2, color=INK, weight="bold", ha="center")
    node_y = [0.637, 0.557, 0.477]
    for row, y in zip(feature_rows, node_y):
        round_box(canvas, 0.258, y, 0.103, 0.058, face="white", edge=PANEL)
        canvas.text(0.267, y + 0.040, f"#{row['rank']}", fontsize=6.4, color=MUTED, va="center")
        canvas.text(0.3095, y + 0.032, f"F{row['feature']}", fontsize=9.0, color=INK,
                    weight="bold", ha="center", va="center")
        canvas.text(0.353, y + 0.014, row["decoder"], fontsize=6.2, color=FORCE,
                    weight="bold", ha="right", va="center")

    round_box(canvas, 0.424, 0.548, 0.108, 0.145, face=ENERGY_LIGHT, edge=ENERGY)
    canvas.text(0.478, 0.667, "ENERGY TARGET", fontsize=7.2, color=ENERGY, weight="bold", ha="center")
    canvas.text(0.478, 0.632, "energy", fontsize=9.7, color=INK, weight="bold", ha="center")
    canvas.text(0.478, 0.606, "clean: joules", fontsize=8.0, color=INK, ha="center")
    canvas.text(0.478, 0.574, "top remains joules", fontsize=7.0, color=MUTED, ha="center")

    arrow(canvas, (0.195, 0.703), (0.242, 0.670), colour=FORCE, linewidth=1.7)
    arrow(canvas, (0.195, 0.500), (0.242, 0.527), colour=MASS, linewidth=1.4, linestyle=(0, (4, 3)))
    arrow(canvas, (0.377, 0.603), (0.424, 0.621), colour=PANEL, linewidth=1.7)

    canvas.text(0.047, 0.383, "Mean newtons-minus-joules gap", fontsize=8.0, color=MUTED, weight="bold")
    gap_rows = [
        ("Clean", clean_gap, MUTED),
        ("Force patch", force_gap, FORCE),
        ("Mass control", mass_gap, MASS),
    ]
    for index, (label, value, colour) in enumerate(gap_rows):
        y = 0.339 - index * 0.052
        canvas.text(0.051, y, label, fontsize=8.2, color=colour, weight="bold", va="center")
        canvas.plot([0.145, 0.493], [y, y], color=GRID, linewidth=1.3)
        x = 0.145 + (value + 12.0) / 12.0 * (0.493 - 0.145)
        canvas.scatter([x], [y], s=48, color=colour, edgecolor="white", linewidth=0.7, zorder=4)
        canvas.text(0.516, y, f"{value:+.2f}", fontsize=8.5, color=colour, weight="bold", ha="right", va="center")
    canvas.plot([0.493, 0.493], [0.224, 0.359], color=INK, linewidth=1.0)
    canvas.text(0.493, 0.203, "0 boundary", fontsize=6.8, color=MUTED, ha="center")

    canvas.text(0.595, 0.835, "B", fontsize=10, color="white", weight="bold", ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.28", "facecolor": INK, "edgecolor": INK})
    canvas.text(0.621, 0.835, "Held-out effect and compactness", fontsize=13, color=INK, weight="bold", va="center")

    effect_axis = figure.add_axes([0.615, 0.535, 0.335, 0.245])
    effects = effect_series(screen)
    y_values = [2, 1, 0]
    effect_axis.axvline(0, color=INK, linewidth=0.9)
    for y, row in zip(y_values, effects):
        low, high = row["ci"]
        effect_axis.plot([low, high], [y, y], color=row["colour"], linewidth=3.0, solid_capstyle="round")
        effect_axis.scatter([row["value"]], [y], s=68, color=row["colour"], edgecolor="white", linewidth=0.8, zorder=4)
        effect_axis.text(high + 0.08, y, f"{row['value']:+.3f}", color=row["colour"], fontsize=8.4,
                         weight="bold", va="center")
    effect_axis.set_yticks(y_values, [row["label"] for row in effects])
    effect_axis.set_xlim(-0.35, 2.75)
    effect_axis.set_ylim(-0.55, 2.55)
    effect_axis.set_xlabel("Change in logit(newtons) - logit(joules)", fontsize=8.0, color=MUTED)
    effect_axis.set_title("Top-3 confirmation (bootstrap 95% CI)", fontsize=9.5, color=INK, weight="bold", loc="left")
    effect_axis.grid(axis="x", color=GRID, linewidth=0.7)
    effect_axis.spines[["top", "right", "left"]].set_visible(False)
    effect_axis.tick_params(axis="y", length=0, labelsize=8.0, colors=INK)
    effect_axis.tick_params(axis="x", labelsize=7.5, colors=MUTED)

    compact_axis = figure.add_axes([0.615, 0.205, 0.335, 0.235])
    labels, values = compactness_series(screen)
    x_values = list(range(len(labels)))
    compact_axis.plot(x_values, values, color=PANEL, linewidth=2.2, marker="o", markersize=6.5)
    compact_axis.fill_between(x_values, values, [0] * len(values), color=PANEL_LIGHT, alpha=0.55)
    compact_axis.axhline(old_effect, color=GOLD, linewidth=1.2, linestyle=(0, (4, 3)),
                         label=f"1k Top-10 ({old_effect:+.3f})")
    compact_axis.scatter([3], [new_effect], s=80, facecolor="white", edgecolor=FORCE, linewidth=2.0, zorder=5)
    compact_axis.annotate(
        f"10k Top-10  {new_effect:+.3f}",
        xy=(3, new_effect),
        xytext=(3.45, new_effect + 0.28),
        fontsize=7.8,
        color=FORCE,
        weight="bold",
        arrowprops={"arrowstyle": "-", "color": FORCE, "linewidth": 0.9},
    )
    compact_axis.set_xticks(x_values, labels)
    compact_axis.set_ylim(0, 3.35)
    compact_axis.set_ylabel("Paired specificity", fontsize=8.0, color=MUTED)
    compact_axis.set_xlabel("Number of selected graph features", fontsize=8.0, color=MUTED)
    compact_axis.set_title("Most of the effect appears by three features", fontsize=9.5, color=INK, weight="bold", loc="left")
    compact_axis.grid(axis="y", color=GRID, linewidth=0.7)
    compact_axis.spines[["top", "right"]].set_visible(False)
    compact_axis.tick_params(labelsize=7.5, colors=MUTED)
    compact_axis.legend(frameon=False, fontsize=7.2, loc="lower right")

    round_box(canvas, 0.595, 0.105, 0.355, 0.060, face=SOFT, edge=GRID)
    canvas.text(0.610, 0.135, "16/16", fontsize=12, color=FORCE, weight="bold", va="center")
    canvas.text(0.662, 0.135, "force effects positive", fontsize=8.0, color=INK, va="center")
    canvas.text(0.790, 0.135, "0/16", fontsize=12, color=ENERGY, weight="bold", va="center")
    canvas.text(0.837, 0.135, "top-token flips", fontsize=8.0, color=INK, va="center")

    canvas.text(
        0.045,
        0.055,
        "Interpretation: the panel transfers a selective force-associated logit signal; it does not replace the model's energy answer.",
        fontsize=9.0,
        color=INK,
        weight="bold",
    )
    canvas.text(
        0.045,
        0.027,
        "Feature ranks were fixed on eight discovery systems and evaluated on sixteen disjoint confirmation systems. Mass is the matched donor control.",
        fontsize=7.7,
        color=MUTED,
    )

    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".png":
            figure.savefig(output_path, dpi=260, bbox_inches="tight", facecolor="white")
        else:
            figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the 10k units force-panel presentation figure")
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("outputs/units_large_data_test/units_large_10000_topk128_force_graph.json"),
    )
    parser.add_argument(
        "--screen",
        type=Path,
        default=Path("outputs/units_large_data_test/units_large_10000_topk128_feature_screen.json"),
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("outputs/units_large_data_test/units_1k_vs_10k_comparison.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("presentation/figures"))
    parser.add_argument("--stem", default="fig_units_10k_force_panel")
    args = parser.parse_args()

    output_paths = [args.output_dir / f"{args.stem}.{suffix}" for suffix in ("png", "pdf", "svg")]
    render(
        load_json(args.graph),
        load_json(args.screen),
        load_json(args.comparison),
        output_paths,
    )
    for path in output_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

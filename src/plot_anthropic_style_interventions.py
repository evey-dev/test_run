"""Draw example-level intervention schematics in Anthropic's visual idiom."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


BACKGROUND = "#FAF9F5"
NODE_FACE = "#E9E8DC"
NODE_EDGE = "#777567"
TEXT = "#191919"
MUTED = "#8B8B87"
GHOST_FACE = "#F8F7F3"
GHOST_EDGE = "#D8D6CF"
GHOST_TEXT = "#B9B8B4"
EDGE = "#633839"
ORANGE = "#D06A00"
ORANGE_LIGHT = "#FFF0E1"
TOKEN_A = "#E8E6DA"
TOKEN_B = "#F1F0EA"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_panel(payload: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next(panel for panel in payload["confirmation"]["panels"] if panel["name"] == name)


def find_case(rows: Sequence[Dict[str, Any]], key: str, value: str) -> Dict[str, Any]:
    return next(row for row in rows if str(row.get(key)) == value)


def panel_axis(figure: Figure, rect: Sequence[float], title: str) -> Axes:
    axis = figure.add_axes(rect)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.add_patch(Rectangle((0, 0), 1, 1, facecolor=BACKGROUND, edgecolor="#E2E0D8", linewidth=1.0))
    axis.text(0.025, 0.965, "GRAPH &\nINTERVENTIONS", fontsize=8.2, color="#6F6D61",
              weight="bold", va="top", linespacing=1.05)
    axis.text(0.245, 0.950, title, fontsize=12.0, color=TEXT, weight="bold", va="top")
    return axis


def node(
    axis: Axes,
    center: tuple[float, float],
    text: str,
    *,
    width: float = 0.27,
    height: float = 0.105,
    ghost: bool = False,
    orange: bool = False,
    fontsize: float = 10.5,
) -> None:
    x = center[0] - width / 2
    y = center[1] - height / 2
    if orange:
        face, edge, colour = ORANGE_LIGHT, ORANGE, TEXT
    elif ghost:
        face, edge, colour = GHOST_FACE, GHOST_EDGE, GHOST_TEXT
    else:
        face, edge, colour = NODE_FACE, NODE_EDGE, TEXT
    axis.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=face, edgecolor=edge, linewidth=1.5,
        )
    )
    axis.text(center[0], center[1], text, ha="center", va="center", fontsize=fontsize,
              color=colour, linespacing=1.05)


def badge(axis: Axes, center: tuple[float, float], text: str) -> None:
    axis.text(
        center[0], center[1], text,
        ha="center", va="center", fontsize=10.0, color="white", weight="bold",
        bbox={"boxstyle": "round,pad=0.28,rounding_size=1.0", "facecolor": ORANGE, "edgecolor": ORANGE},
        zorder=8,
    )


def line_arrow(
    axis: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = EDGE,
    alpha: float = 1.0,
    linewidth: float = 1.8,
    orange: bool = False,
) -> None:
    actual_color = ORANGE if orange else color
    axis.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=14,
            linewidth=linewidth, color=actual_color, alpha=alpha,
            shrinkA=0, shrinkB=0, zorder=3,
        )
    )


def elbow_arrow(
    axis: Axes,
    points: Sequence[tuple[float, float]],
    *,
    color: str = EDGE,
    alpha: float = 1.0,
    linewidth: float = 1.8,
    orange: bool = False,
) -> None:
    actual_color = ORANGE if orange else color
    for start, end in zip(points[:-2], points[1:-1]):
        axis.plot([start[0], end[0]], [start[1], end[1]], color=actual_color,
                  alpha=alpha, linewidth=linewidth, solid_capstyle="round", zorder=2)
    line_arrow(axis, points[-2], points[-1], color=actual_color, alpha=alpha,
               linewidth=linewidth, orange=False)


def token_strip(axis: Axes, labels: Sequence[str], widths: Sequence[float] | None = None) -> None:
    x0, y0, total_width, height = 0.035, 0.035, 0.93, 0.065
    if widths is None:
        widths = [1 / len(labels)] * len(labels)
    scale = total_width / sum(widths)
    x = x0
    for index, (label, raw_width) in enumerate(zip(labels, widths)):
        width = raw_width * scale
        axis.add_patch(Rectangle((x, y0), width, height, facecolor=TOKEN_A if index % 2 == 0 else TOKEN_B,
                                 edgecolor="none"))
        axis.text(x + width / 2, y0 + height / 2, label, fontsize=9.3, color=TEXT,
                  ha="center", va="center", family="monospace")
        x += width


def completion_strip(
    axis: Axes,
    prefix: str,
    outcome: str,
    *,
    toward: bool,
    detail: str | None = None,
) -> None:
    x0, y0, width, height = 0.035, 0.835, 0.93, 0.075
    right_width = 0.34
    left_width = width - right_width
    axis.add_patch(Rectangle((x0, y0), left_width, height, facecolor=TOKEN_B, edgecolor="none"))
    axis.add_patch(Rectangle((x0 + left_width, y0), right_width, height,
                             facecolor=ORANGE_LIGHT if toward else NODE_FACE,
                             edgecolor=ORANGE if toward else "none",
                             linewidth=1.4))
    axis.plot([x0, x0 + width], [y0, y0], color="#BDBBB3", linewidth=0.8)
    axis.text(x0 + 0.02, y0 + height / 2, prefix, fontsize=9.8, color=TEXT,
              va="center", family="monospace")
    outcome_x = x0 + left_width + right_width / 2
    if detail is None:
        axis.text(outcome_x, y0 + height / 2, outcome, fontsize=10.2, color=TEXT,
                  va="center", ha="center", weight="bold")
    else:
        axis.text(outcome_x, y0 + height * 0.64, outcome, fontsize=9.3, color=TEXT,
                  va="center", ha="center", weight="bold")
        axis.text(outcome_x, y0 + height * 0.25, detail, fontsize=8.2,
                  color=ORANGE if toward else MUTED, va="center", ha="center",
                  family="monospace")


def effect_label(axis: Axes, x: float, y: float, text: str, *, color: str = MUTED) -> None:
    axis.text(x, y, text, fontsize=8.2, color=color, ha="center", va="center", linespacing=1.1)


def save(figure: Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 280} if suffix == "png" else {}
        figure.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(figure)


def math_figure(payload: Dict[str, Any], output_dir: Path) -> None:
    panel = find_panel(payload, "top_10_primary")
    carry = find_case(panel["case_effects"], "case_key", "66+83->68+83")

    figure = plt.figure(figsize=(12.0, 5.8), facecolor="white")
    target = panel_axis(figure, [0.035, 0.08, 0.455, 0.84], "CARRY TARGET")
    control = panel_axis(figure, [0.510, 0.08, 0.455, 0.84], "MATCHED NO-CARRY CONTROL")

    completion_strip(
        target,
        "68 + 83 = 1",
        'towards "4"',
        toward=True,
        detail=f"delta[5-4] = {carry['target_delta']:+.3f}",
    )
    node(target, (0.66, 0.690), 'Say "5"', ghost=True, width=0.25)
    node(target, (0.50, 0.435), "Carry-associated\nTop-10 panel", width=0.31,
         height=0.125, ghost=True)
    badge(target, (0.29, 0.505), "0x")
    node(target, (0.25, 0.225), "ones digits\n8 + 3", width=0.25)
    node(target, (0.68, 0.225), "answer prefix\n1", width=0.25)
    elbow_arrow(target, [(0.25, 0.278), (0.25, 0.325), (0.43, 0.325), (0.43, 0.373)])
    elbow_arrow(target, [(0.68, 0.278), (0.68, 0.325), (0.57, 0.325), (0.57, 0.373)])
    elbow_arrow(target, [(0.50, 0.498), (0.50, 0.555), (0.66, 0.555), (0.66, 0.635)], alpha=0.25)
    line_arrow(target, (0.66, 0.745), (0.66, 0.830), alpha=0.30)
    token_strip(target, ["68", "+", "83", "Answer: 1"], [1.0, 0.45, 1.0, 1.6])

    completion_strip(
        control,
        "66 + 83 = 1",
        '"4" unchanged',
        toward=False,
        detail=f"delta[4-5] = {carry['control_delta']:+.3f}",
    )
    node(control, (0.66, 0.690), 'Say "4"', width=0.25)
    node(control, (0.50, 0.435), "same feature IDs\nTop-10 panel", width=0.31, height=0.125, ghost=True)
    badge(control, (0.29, 0.505), "0x")
    node(control, (0.25, 0.225), "ones digits\n6 + 3", width=0.25)
    node(control, (0.68, 0.225), "answer prefix\n1", width=0.25)
    elbow_arrow(control, [(0.25, 0.278), (0.25, 0.325), (0.43, 0.325), (0.43, 0.373)], alpha=0.28)
    elbow_arrow(control, [(0.68, 0.278), (0.68, 0.325), (0.57, 0.325), (0.57, 0.373)], alpha=0.28)
    elbow_arrow(control, [(0.50, 0.498), (0.50, 0.555), (0.66, 0.555), (0.66, 0.635)], alpha=0.25)
    line_arrow(control, (0.66, 0.745), (0.66, 0.830), alpha=0.85)
    token_strip(control, ["66", "+", "83", "Answer: 1"], [1.0, 0.45, 1.0, 1.6])

    figure.text(0.5, 0.025,
                f"Held-out example from the frozen graph Top-10 test | case-specific paired difference {carry['paired_difference']:+.3f}",
                ha="center", fontsize=8.8, color=MUTED)
    save(figure, output_dir, "fig_anthropic_style_math_carry_inhibition")


def capitals_figure(payload: Dict[str, Any], output_dir: Path) -> None:
    panel = find_panel(payload, "top_3_primary")
    case = find_case(panel["case_effects"], "country", "Armenia")

    figure = plt.figure(figsize=(12.0, 5.8), facecolor="white")
    target = panel_axis(figure, [0.035, 0.08, 0.455, 0.84], "CAPITAL-RELATION TARGET")
    control = panel_axis(figure, [0.510, 0.08, 0.455, 0.84], "INVERSE COUNTRY CONTROL")

    completion_strip(
        target,
        "capital ... Gyumri?",
        "toward Armenia",
        toward=True,
        detail=f"delta[cap-country] = {case['capital_prompt_delta']:+.3f}",
    )
    node(target, (0.66, 0.690), "Say Yerevan", ghost=True, width=0.27)
    node(target, (0.50, 0.435), "Capital relation\nL28 Top-3", width=0.30,
         height=0.125, ghost=True)
    badge(target, (0.29, 0.505), "0x")
    node(target, (0.25, 0.225), "capital of\ncountry containing", width=0.28, fontsize=9.2)
    node(target, (0.68, 0.225), "Gyumri", width=0.23)
    elbow_arrow(target, [(0.25, 0.278), (0.25, 0.325), (0.43, 0.325), (0.43, 0.373)],
                alpha=0.28)
    elbow_arrow(target, [(0.68, 0.278), (0.68, 0.325), (0.57, 0.325), (0.57, 0.373)],
                alpha=0.28)
    elbow_arrow(target, [(0.50, 0.498), (0.50, 0.555), (0.66, 0.555), (0.66, 0.635)],
                alpha=0.22)
    line_arrow(target, (0.66, 0.745), (0.66, 0.830), alpha=0.30)
    token_strip(target, ["capital relation", "Gyumri", "?"], [1.8, 1.0, 0.35])

    completion_strip(
        control,
        "country ... Gyumri?",
        "Armenia unchanged",
        toward=False,
        detail=f"delta[cap-country] = {case['inverse_country_prompt_delta']:+.3f}",
    )
    node(control, (0.66, 0.690), "Say Armenia", width=0.27)
    node(control, (0.50, 0.435), "same feature IDs\nL28 Top-3", width=0.30, height=0.125, ghost=True)
    badge(control, (0.29, 0.505), "0x")
    node(control, (0.25, 0.225), "country\ncontaining", width=0.26)
    node(control, (0.68, 0.225), "Gyumri", width=0.23)
    elbow_arrow(control, [(0.25, 0.278), (0.25, 0.325), (0.43, 0.325), (0.43, 0.373)],
                alpha=0.28)
    elbow_arrow(control, [(0.68, 0.278), (0.68, 0.325), (0.57, 0.325), (0.57, 0.373)],
                alpha=0.28)
    elbow_arrow(control, [(0.50, 0.498), (0.50, 0.555), (0.66, 0.555), (0.66, 0.635)],
                alpha=0.22)
    line_arrow(control, (0.66, 0.745), (0.66, 0.830), alpha=0.85)
    token_strip(control, ["country relation", "Gyumri", "?"], [1.8, 1.0, 0.35])

    figure.text(0.5, 0.025,
                f"Held-out Armenia example | case-specific relation effect {case['relation_specific_difference']:+.3f}",
                ha="center", fontsize=8.8, color=MUTED)
    save(figure, output_dir, "fig_anthropic_style_capitals_relation_inhibition")


def capitals_figure_v2(payload: Dict[str, Any], output_dir: Path) -> None:
    panel = find_panel(payload, "top_3_primary")
    case = find_case(panel["case_effects"], "country", "Armenia")

    figure = plt.figure(figsize=(12.0, 5.8), facecolor="white")
    target = panel_axis(figure, [0.035, 0.08, 0.455, 0.84], "CAPITAL-RELATION TARGET")
    control = panel_axis(figure, [0.510, 0.08, 0.455, 0.84], "INVERSE COUNTRY CONTROL")

    completion_strip(
        target,
        "capital ... Gyumri?",
        "toward Armenia",
        toward=True,
        detail=f"delta[cap-country] = {case['capital_prompt_delta']:+.3f}",
    )
    node(target, (0.68, 0.690), "Say Yerevan", ghost=True, width=0.27)
    node(target, (0.60, 0.490), "Capital relation\nL28 Top-3", width=0.30,
         height=0.120, ghost=True)
    badge(target, (0.405, 0.535), "0x")
    node(target, (0.30, 0.325), "Country containing\nGyumri", width=0.29,
         height=0.105, fontsize=9.3)
    node(target, (0.15, 0.160), "country", width=0.19, height=0.095)
    node(target, (0.42, 0.160), "Gyumri", width=0.19, height=0.095)
    node(target, (0.76, 0.160), "capital", width=0.19, height=0.095)

    elbow_arrow(target, [(0.15, 0.208), (0.15, 0.240), (0.245, 0.240), (0.245, 0.272)])
    elbow_arrow(target, [(0.42, 0.208), (0.42, 0.240), (0.355, 0.240), (0.355, 0.272)])
    elbow_arrow(target, [(0.30, 0.378), (0.30, 0.405), (0.52, 0.405), (0.52, 0.430)],
                alpha=0.28)
    elbow_arrow(target, [(0.76, 0.208), (0.76, 0.405), (0.68, 0.405), (0.68, 0.430)],
                alpha=0.28)
    elbow_arrow(target, [(0.60, 0.550), (0.60, 0.585), (0.68, 0.585), (0.68, 0.635)],
                alpha=0.22)
    line_arrow(target, (0.68, 0.745), (0.68, 0.830), alpha=0.30)
    token_strip(target, ["capital relation", "Gyumri", "?"], [1.8, 1.0, 0.35])

    completion_strip(
        control,
        "country ... Gyumri?",
        "Armenia unchanged",
        toward=False,
        detail=f"delta[cap-country] = {case['inverse_country_prompt_delta']:+.3f}",
    )
    node(control, (0.60, 0.690), "Say Armenia", width=0.27)
    node(control, (0.82, 0.420), "same capital IDs\nL28 Top-3", width=0.25,
         height=0.115, ghost=True, fontsize=9.5)
    badge(control, (0.665, 0.455), "0x")
    node(control, (0.22, 0.200), "country", width=0.21)
    node(control, (0.60, 0.200), "Gyumri", width=0.21)

    elbow_arrow(control, [(0.22, 0.253), (0.22, 0.355), (0.50, 0.355), (0.50, 0.635)])
    line_arrow(control, (0.60, 0.253), (0.60, 0.635))
    elbow_arrow(control, [(0.82, 0.478), (0.82, 0.555), (0.70, 0.555), (0.70, 0.635)],
                alpha=0.22)
    line_arrow(control, (0.60, 0.745), (0.60, 0.830), alpha=0.85)
    token_strip(control, ["country relation", "Gyumri", "?"], [1.8, 1.0, 0.35])

    figure.text(0.5, 0.025,
                f"Held-out Armenia example | case-specific relation effect {case['relation_specific_difference']:+.3f}",
                ha="center", fontsize=8.8, color=MUTED)
    save(figure, output_dir, "fig_anthropic_style_capitals_relation_inhibition_v2")


def units_figure(payload: Dict[str, Any], output_dir: Path) -> None:
    panel = find_panel(payload, "top_10_primary")
    case = find_case(panel["case_effects"], "context",
                     "railway brake linkage under steady operating conditions")

    figure = plt.figure(figsize=(12.0, 5.8), facecolor="white")
    target = panel_axis(figure, [0.035, 0.08, 0.455, 0.84], "FORCE DONOR -> ENERGY TARGET")
    control = panel_axis(figure, [0.510, 0.08, 0.455, 0.84], "MASS DONOR CONTROL")

    completion_strip(
        target,
        "energy SI unit?",
        "towards newtons",
        toward=True,
        detail=f"delta[new-j] = {case['force_source_delta']:+.3f}",
    )
    node(target, (0.67, 0.690), "Say joules", ghost=True, width=0.25)
    node(target, (0.67, 0.435), "Energy target\nsame coordinates", width=0.29, height=0.125)
    node(target, (0.27, 0.435), "Force-source\nTop-10 values", width=0.28, height=0.125, orange=True)
    badge(target, (0.47, 0.520), "SWAP")
    node(target, (0.27, 0.225), "railway brake\nforce prompt", width=0.28)
    node(target, (0.67, 0.225), "railway brake\nenergy prompt", width=0.28)
    line_arrow(target, (0.27, 0.278), (0.27, 0.370), orange=True)
    line_arrow(target, (0.67, 0.278), (0.67, 0.370), alpha=0.55)
    line_arrow(target, (0.41, 0.435), (0.515, 0.435), orange=True, linewidth=2.2)
    line_arrow(target, (0.67, 0.498), (0.67, 0.635), orange=True, linewidth=2.1)
    line_arrow(target, (0.67, 0.745), (0.67, 0.830), orange=True, linewidth=2.1)
    token_strip(target, ["railway brake", "energy", "unit"], [1.7, 1.0, 0.7])

    completion_strip(
        control,
        "energy SI unit?",
        "joules unchanged",
        toward=False,
        detail=f"delta[new-j] = {case['mass_source_delta']:+.3f}",
    )
    node(control, (0.67, 0.690), "Say joules", width=0.25)
    node(control, (0.67, 0.435), "Energy target\nsame coordinates", width=0.29, height=0.125)
    node(control, (0.27, 0.435), "Mass-source\nsame feature IDs", width=0.28, height=0.125)
    badge(control, (0.47, 0.520), "SWAP")
    node(control, (0.27, 0.225), "railway brake\nmass prompt", width=0.28)
    node(control, (0.67, 0.225), "railway brake\nenergy prompt", width=0.28)
    line_arrow(control, (0.27, 0.278), (0.27, 0.370))
    line_arrow(control, (0.67, 0.278), (0.67, 0.370), alpha=0.55)
    line_arrow(control, (0.41, 0.435), (0.515, 0.435), orange=True, linewidth=2.0)
    line_arrow(control, (0.67, 0.498), (0.67, 0.635), alpha=0.70)
    line_arrow(control, (0.67, 0.745), (0.67, 0.830), alpha=0.85)
    token_strip(control, ["railway brake", "energy", "unit"], [1.7, 1.0, 0.7])

    figure.text(0.5, 0.025,
                f"Held-out railway-brake example | force-minus-mass specificity {case['force_minus_mass_difference']:+.3f}",
                ha="center", fontsize=8.8, color=MUTED)
    save(figure, output_dir, "fig_anthropic_style_units_force_swap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math", type=Path,
                        default=Path("outputs/math_large_data_test/math_large_10000_topk256_graph_feature_screen.json"))
    parser.add_argument("--capitals", type=Path,
                        default=Path("outputs/capitals_large_data_test/capitals_large_10000_topk128_relation_screen.json"))
    parser.add_argument("--units", type=Path,
                        default=Path("outputs/units_large_data_test/units_large_10000_topk128_feature_screen.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("presentation/figures"))
    args = parser.parse_args()

    math_figure(load_json(args.math), args.output_dir)
    capitals_payload = load_json(args.capitals)
    capitals_figure(capitals_payload, args.output_dir)
    capitals_figure_v2(capitals_payload, args.output_dir)
    units_figure(load_json(args.units), args.output_dir)
    print(f"Saved Anthropic-style intervention figures under {args.output_dir}")


if __name__ == "__main__":
    main()

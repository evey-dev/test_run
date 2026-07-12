"""Plot the balanced arithmetic carry localisation and frozen replication.

The preferred inputs are the completed localisation and replication JSON files.
When those Drive artifacts have not been copied into the checkout, the renderer
can recover the same aggregate values from the saved outputs in the completed
notebook. The notebook fallback verifies the relevant output structure instead
of silently substituting illustrative values.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


BACKGROUND = "#FFFFFF"
INK = "#17212B"
MUTED = "#5F6B76"
GRID = "#D8DEE6"
PRIMARY = "#718096"
EXPLORATORY = "#B7791F"
REPLICATION = "#17847A"
TARGET = "#C53B4C"
CONTROL = "#2F6FA3"
LAYER_24 = "#7B5AA6"
LAYER_28 = "#C45A72"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def panel_by_name(source: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next(
        panel
        for panel in source["causal_confirmation"]["panels"]
        if panel["name"] == name
    )


def summary_from_json(source: Dict[str, Any], replication: Dict[str, Any]) -> Dict[str, Any]:
    if source.get("status") != "complete" or replication.get("status") != "complete":
        raise ValueError("Both arithmetic result JSON files must be complete")
    primary = source["primary_result"]["causal_confirmation"]
    exploratory = panel_by_name(source, "top_20")["causal_summary"]
    replicated = replication["primary_result"]["causal_summary"]
    return {
        "raw_auc": {
            int(row["layer"]): float(row["confirmation"]["output_digit_conditioned_auc"])
            for row in source["raw_mlp_localisation"]
        },
        "features": [row["key"] for row in source["sae_feature_discovery"]["frozen_top_20"]],
        "top20_activation": float(
            source["panel_activation_validation"]["top_20"]["confirmation"][
                "mean_within_digit_carry_minus_no_carry"
            ]
        ),
        "primary": primary,
        "exploratory": exploratory,
        "replication": replicated,
        "replication_passed": bool(
            replication["primary_result"]["replicates_frozen_top20_carry_selectivity"]
        ),
    }


def cell_by_source(notebook: Dict[str, Any], marker: str) -> Dict[str, Any]:
    return next(
        cell
        for cell in notebook["cells"]
        if marker in "".join(cell.get("source", []))
    )


def stream_text(cell: Dict[str, Any]) -> str:
    return "\n".join(
        "".join(output.get("text", []))
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def display_texts(cell: Dict[str, Any]) -> List[str]:
    return [
        "".join(output.get("data", {}).get("text/plain", []))
        for output in cell.get("outputs", [])
        if "text/plain" in output.get("data", {})
    ]


def parse_json_from_stream(text: str) -> Dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError("Could not find the primary-result JSON in the notebook output")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    return value


def parse_scalar(text: str, label: str) -> float:
    match = re.search(rf"(?m)^{re.escape(label)}\s+(-?\d+(?:\.\d+)?)\s*$", text)
    if not match:
        raise ValueError(f"Could not parse {label!r} from notebook output")
    return float(match.group(1))


def parse_interval(text: str, label: str) -> List[float]:
    match = re.search(
        rf"(?m)^{re.escape(label)}\s+\[(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\]\s*$",
        text,
    )
    if not match:
        raise ValueError(f"Could not parse {label!r} from notebook output")
    return [float(match.group(1)), float(match.group(2))]


def summary_from_notebook(path: Path) -> Dict[str, Any]:
    notebook = load_json(path)

    run_cell = cell_by_source(notebook, "src.math_carry_balanced_localization")
    primary_result = parse_json_from_stream(stream_text(run_cell))
    if primary_result.get("supports_compact_carry_selectivity_under_predeclared_rule"):
        raise ValueError("Notebook no longer records the expected failed Top-10 primary")

    inspection_cell = cell_by_source(notebook, "raw_table = pd.DataFrame")
    inspection_outputs = display_texts(inspection_cell)
    if len(inspection_outputs) < 2:
        raise ValueError("Notebook inspection cell is missing its recorded tables")
    raw_rows = re.findall(
        r"(?m)^\s*(4|8|12|16|20|24|28)\s+([01]\.\d+)\s+([01]\.\d+)\s*$",
        inspection_outputs[0],
    )
    raw_auc = {int(layer): float(confirmation) for layer, _, confirmation in raw_rows[:7]}
    if sorted(raw_auc) != [4, 8, 12, 16, 20, 24, 28]:
        raise ValueError("Could not recover all seven raw-MLP confirmation AUC values")

    features: List[str] = []
    for key in re.findall(r"L(?:4|8|12|16|20|24|28)F\d+", inspection_outputs[1]):
        if key not in features:
            features.append(key)
    features = features[:20]
    if len(features) != 20:
        raise ValueError("Could not recover the frozen Top-20 feature IDs")

    panel_cell = cell_by_source(notebook, "panel_table = pd.DataFrame")
    panel_outputs = display_texts(panel_cell)
    if not panel_outputs:
        raise ValueError("Notebook panel table has no recorded output")
    panel_text = panel_outputs[0]
    activation_match = re.search(r"(?m)^\s*4\s+20\s+(-?\d+\.\d+)\s*$", panel_text)
    if not activation_match:
        raise ValueError("Could not recover the Top-20 activation confirmation")
    causal_section = panel_text.split("no-carry control delta", maxsplit=1)
    if len(causal_section) != 2:
        raise ValueError("Could not find the causal columns in the panel table")
    exploratory_match = re.search(
        r"(?m)^\s*4\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+"
        r"\[(-?\d+\.\d+),\s*(-?\d+\.\d+)\]\s*$",
        causal_section[1],
    )
    if not exploratory_match:
        raise ValueError("Could not recover the exploratory Top-20 causal row")

    replication_cell = cell_by_source(notebook, "replication = json.loads")
    replication_outputs = display_texts(replication_cell)
    if not replication_outputs:
        raise ValueError("Notebook replication cell has no recorded summary")
    replication_text = replication_outputs[0]
    replication_stream = stream_text(replication_cell)
    if "Frozen Top-20 replication passed: True" not in replication_stream:
        raise ValueError("Notebook does not record a passing frozen Top-20 replication")

    primary = primary_result["causal_confirmation"]
    exploratory = {
        "eligible_pairs": 32,
        "mean_no_carry_control_delta": float(exploratory_match.group(1)),
        "mean_paired_difference": float(exploratory_match.group(2)),
        "bootstrap_95_ci_mean_paired_difference": [
            float(exploratory_match.group(3)),
            float(exploratory_match.group(4)),
        ],
    }
    replicated = {
        "eligible_pairs": int(parse_scalar(replication_text, "eligible_pairs")),
        "mean_carry_target_delta": parse_scalar(
            replication_text, "mean_carry_target_delta"
        ),
        "bootstrap_95_ci_mean_carry_target_delta": parse_interval(
            replication_text, "bootstrap_95_ci_mean_carry_target_delta"
        ),
        "mean_no_carry_control_delta": parse_scalar(
            replication_text, "mean_no_carry_control_delta"
        ),
        "bootstrap_95_ci_mean_no_carry_control_delta": parse_interval(
            replication_text, "bootstrap_95_ci_mean_no_carry_control_delta"
        ),
        "mean_paired_difference": parse_scalar(
            replication_text, "mean_paired_difference"
        ),
        "bootstrap_95_ci_mean_paired_difference": parse_interval(
            replication_text, "bootstrap_95_ci_mean_paired_difference"
        ),
        "fraction_carry_more_negative_than_control": parse_scalar(
            replication_text, "fraction_carry_more_negative_than_control"
        ),
        "carry_top_prediction_transfer_fraction": parse_scalar(
            replication_text, "carry_top_prediction_transfer_fraction"
        ),
    }
    return {
        "raw_auc": raw_auc,
        "features": features,
        "top20_activation": float(activation_match.group(1)),
        "primary": primary,
        "exploratory": exploratory,
        "replication": replicated,
        "replication_passed": True,
    }


def asymmetric_errors(mean: float, interval: Sequence[float]) -> List[List[float]]:
    return [[mean - float(interval[0])], [float(interval[1]) - mean]]


def add_panel_label(axis, label: str) -> None:
    axis.text(
        -0.11,
        1.05,
        label,
        transform=axis.transAxes,
        fontsize=11,
        weight="bold",
        color=INK,
        va="top",
    )


def chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def render(data: Dict[str, Any], output_paths: Sequence[Path]) -> None:
    if not data["replication_passed"]:
        raise ValueError("The positive replication figure requires a passing frozen test")
    features = list(data["features"])
    if len(features) != 20:
        raise ValueError(f"Expected 20 frozen features, found {len(features)}")

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
    from matplotlib.patches import FancyBboxPatch

    figure = plt.figure(figsize=(12.4, 7.3), facecolor=BACKGROUND)
    grid = figure.add_gridspec(
        2,
        2,
        left=0.13,
        right=0.975,
        bottom=0.13,
        top=0.82,
        hspace=0.43,
        wspace=0.28,
    )
    auc_axis = figure.add_subplot(grid[0, 0])
    feature_axis = figure.add_subplot(grid[0, 1])
    sequence_axis = figure.add_subplot(grid[1, 0])
    replication_axis = figure.add_subplot(grid[1, 1])

    figure.text(
        0.05,
        0.948,
        "A frozen late-layer SAE panel shows replicated carry-selective logit control",
        fontsize=17.0,
        weight="bold",
        color=INK,
        va="top",
    )
    figure.text(
        0.05,
        0.905,
        "The Top-10 primary was null; the exploratory Top-20 panel was frozen and tested once on intervention-untouched cases",
        fontsize=10.2,
        color=MUTED,
        va="top",
    )

    layers = sorted(data["raw_auc"])
    auc_values = [data["raw_auc"][layer] for layer in layers]
    auc_axis.plot(layers, auc_values, color=CONTROL, marker="o", linewidth=2.0, markersize=5.5)
    auc_axis.axhline(0.5, color=PRIMARY, linestyle="--", linewidth=1.1)
    auc_axis.fill_between([22, 30], 0.28, 1.03, color="#E4F3F0", alpha=0.8, zorder=0)
    auc_axis.set_xlim(2.5, 29.5)
    auc_axis.set_ylim(0.28, 1.03)
    auc_axis.set_xticks(layers)
    auc_axis.set_xlabel("Transformer layer")
    auc_axis.set_ylabel("Conditioned confirmation AUC")
    auc_axis.set_title("Raw MLP carry decodability", loc="left", fontsize=11.5, weight="bold")
    auc_axis.text(24.0, 0.94, "late-layer\nsignal", color=REPLICATION, fontsize=8.5, va="top")
    auc_axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    auc_axis.spines[["top", "right"]].set_visible(False)
    add_panel_label(auc_axis, "a")

    feature_axis.set_xlim(0, 1)
    feature_axis.set_ylim(0, 1)
    feature_axis.axis("off")
    feature_axis.set_title("Frozen carry-associated panel (20 latents)", loc="left", fontsize=11.5, weight="bold")
    add_panel_label(feature_axis, "b")
    grouped = {
        24: [key.split("F", maxsplit=1)[1] for key in features if key.startswith("L24F")],
        28: [key.split("F", maxsplit=1)[1] for key in features if key.startswith("L28F")],
    }
    y_positions = {24: 0.72, 28: 0.39}
    colours = {24: LAYER_24, 28: LAYER_28}
    for layer in (24, 28):
        y = y_positions[layer]
        feature_axis.add_patch(
            FancyBboxPatch(
                (0.02, y - 0.13),
                0.96,
                0.27,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                facecolor="white",
                edgecolor=colours[layer],
                linewidth=1.3,
            )
        )
        feature_axis.text(0.055, y + 0.055, f"Layer {layer}", weight="bold", color=colours[layer], fontsize=9.2)
        lines = [", ".join(f"F{feature}" for feature in row) for row in chunks(grouped[layer], 6)]
        feature_axis.text(0.055, y + 0.005, "\n".join(lines), color=INK, fontsize=8.0, va="top", linespacing=1.25)
        feature_axis.text(0.945, y + 0.055, f"{len(grouped[layer])} features", ha="right", color=MUTED, fontsize=8.1)
    feature_axis.text(
        0.02,
        0.205,
        "32 discovery pairs  |  output-digit-conditioned ranking  |  "
        f"held-out difference +{data['top20_activation']:.3f}",
        fontsize=7.6,
        color=MUTED,
        va="center",
    )
    feature_axis.add_patch(
        FancyBboxPatch(
            (0.02, 0.015),
            0.96,
            0.13,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#F9E7EA",
            edgecolor=TARGET,
            linewidth=1.3,
        )
    )
    feature_axis.text(0.065, 0.08, "X", color=TARGET, fontsize=14, weight="bold", ha="center", va="center")
    feature_axis.text(0.115, 0.102, "INHIBIT THE FROZEN PANEL", color=TARGET, fontsize=8.1, weight="bold", va="center")
    feature_axis.text(
        0.115,
        0.055,
        r"$z_S \leftarrow 0$ at the final token; clean reconstruction residual preserved",
        color=INK,
        fontsize=7.7,
        va="center",
    )

    primary = data["primary"]
    exploratory = data["exploratory"]
    replicated = data["replication"]
    sequence_labels = ["Top-10 primary (null)", "Top-20 exploratory", "Top-20 replication"]
    sequence_means = [
        float(primary["mean_paired_difference"]),
        float(exploratory["mean_paired_difference"]),
        float(replicated["mean_paired_difference"]),
    ]
    sequence_cis = [
        primary["bootstrap_95_ci_mean_paired_difference"],
        exploratory["bootstrap_95_ci_mean_paired_difference"],
        replicated["bootstrap_95_ci_mean_paired_difference"],
    ]
    sequence_colours = [PRIMARY, EXPLORATORY, REPLICATION]
    sequence_y = [2, 1, 0]
    sequence_axis.axvspan(-0.18, 0, color="#E4F3F0", alpha=0.55, zorder=0)
    sequence_axis.axvline(0, color=INK, linewidth=1.0)
    for y, mean, interval, colour in zip(
        sequence_y, sequence_means, sequence_cis, sequence_colours
    ):
        sequence_axis.errorbar(
            mean,
            y,
            xerr=asymmetric_errors(mean, interval),
            fmt="o",
            color=colour,
            ecolor=colour,
            elinewidth=2.0,
            capsize=4,
            markersize=7,
            zorder=3,
        )
        sequence_axis.text(float(interval[1]) + 0.008, y, f"{mean:+.3f}", va="center", color=colour, fontsize=8.3)
    sequence_axis.set_yticks(sequence_y, sequence_labels)
    sequence_axis.set_xlim(-0.18, 0.09)
    sequence_axis.set_ylim(-0.6, 2.6)
    sequence_axis.set_xlabel("Carry minus no-carry gap delta (logits; negative is selective)")
    sequence_axis.set_title("From null primary to frozen replication", loc="left", fontsize=11.5, weight="bold")
    sequence_axis.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)
    sequence_axis.spines[["top", "right", "left"]].set_visible(False)
    sequence_axis.tick_params(axis="y", length=0)
    add_panel_label(sequence_axis, "c")

    component_labels = ["Carry target", "No-carry control", "Paired specificity"]
    component_means = [
        float(replicated["mean_carry_target_delta"]),
        float(replicated["mean_no_carry_control_delta"]),
        float(replicated["mean_paired_difference"]),
    ]
    component_cis = [
        replicated["bootstrap_95_ci_mean_carry_target_delta"],
        replicated["bootstrap_95_ci_mean_no_carry_control_delta"],
        replicated["bootstrap_95_ci_mean_paired_difference"],
    ]
    component_colours = [TARGET, CONTROL, REPLICATION]
    component_y = [2, 1, 0]
    replication_axis.axvline(0, color=INK, linewidth=1.0)
    for y, mean, interval, colour in zip(
        component_y, component_means, component_cis, component_colours
    ):
        replication_axis.errorbar(
            mean,
            y,
            xerr=asymmetric_errors(mean, interval),
            fmt="o",
            color=colour,
            ecolor=colour,
            elinewidth=2.0,
            capsize=4,
            markersize=7,
            zorder=3,
        )
        replication_axis.text(float(interval[1]) + 0.007, y, f"{mean:+.3f}", va="center", color=colour, fontsize=8.3)
    replication_axis.set_yticks(component_y, component_labels)
    replication_axis.set_xlim(-0.16, 0.085)
    replication_axis.set_ylim(-0.6, 2.6)
    replication_axis.set_xlabel("Correct-minus-dropped-carry gap delta (logits)")
    replication_axis.set_title("Independent untouched-case test ($n=32$)", loc="left", fontsize=11.5, weight="bold")
    replication_axis.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)
    replication_axis.spines[["top", "right", "left"]].set_visible(False)
    replication_axis.tick_params(axis="y", length=0)
    add_panel_label(replication_axis, "d")

    fraction = 100.0 * float(replicated["fraction_carry_more_negative_than_control"])
    flips = float(replicated["carry_top_prediction_transfer_fraction"])
    figure.text(
        0.05,
        0.052,
        f"Replication criterion passed: {fraction:.1f}% of pairs had a more negative carry effect; top-token transfer fraction = {flips:.1f}.",
        fontsize=9.0,
        color=INK,
        weight="bold",
    )
    figure.text(
        0.05,
        0.024,
        "Interpretation: a distributed carry-associated causal panel at the tested MLP sites, not a monosemantic carry bit, answer flip, or complete addition circuit.",
        fontsize=8.6,
        color=MUTED,
    )

    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=220, facecolor=figure.get_facecolor())
        if output_path.suffix == ".svg":
            svg = output_path.read_text(encoding="utf-8")
            output_path.write_text(
                "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                encoding="utf-8",
            )
        print(f"Saved {output_path}")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot balanced carry localisation and independent Top-20 replication"
    )
    parser.add_argument(
        "--source",
        default=(
            "outputs/math_carry_localization/"
            "math_topk256_balanced_carry_localization.json"
        ),
    )
    parser.add_argument(
        "--replication",
        default=(
            "outputs/math_carry_localization/"
            "math_topk256_balanced_top20_replication.json"
        ),
    )
    parser.add_argument(
        "--notebook",
        default="run_gpu_math_carry_balanced_localization.ipynb",
        help="Fallback source when the Drive JSON artifacts are not in this checkout",
    )
    parser.add_argument(
        "--output-prefix",
        default="report/figures/fig_math_balanced_carry_replication",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    replication_path = Path(args.replication)
    if source_path.exists() and replication_path.exists():
        print(f"Reading {source_path} and {replication_path}")
        data = summary_from_json(load_json(source_path), load_json(replication_path))
    else:
        notebook_path = Path(args.notebook)
        print(
            "Result JSON files are not both present; reading exact recorded aggregates from "
            f"{notebook_path}"
        )
        data = summary_from_notebook(notebook_path)

    prefix = Path(args.output_prefix)
    render(data, [prefix.with_suffix(".pdf"), prefix.with_suffix(".png"), prefix.with_suffix(".svg")])


if __name__ == "__main__":
    main()

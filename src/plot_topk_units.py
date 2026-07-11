"""Plot units TopK SAE selection and force-specific confirmation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


LAYER_COLOURS = {
    4: "#4C78A8",
    8: "#72B7B2",
    12: "#54A24B",
    16: "#ECA82C",
    20: "#F58518",
    24: "#E45756",
    28: "#B279A2",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_diagnostics(path: Path) -> Dict[str, float]:
    rows = read_csv(path)
    return {
        "mean_fve": mean(float(row["validation_fraction_variance_explained"]) for row in rows),
        "mean_l0": mean(float(row["validation_mean_l0"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot units TopK selection and confirmation")
    parser.add_argument("--candidate-diagnostics", nargs="+", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--original-diagnostics", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/topk_units_retrain/figures"))
    parser.add_argument("--stem", default="fig_units_topk_selection_confirmation")
    args = parser.parse_args()

    with args.selection.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    selected = selection.get("selected")
    if selected is None:
        raise ValueError("Selection JSON contains no eligible candidate")
    points = []
    for path in args.candidate_diagnostics:
        candidate = next(
            row
            for row in selection["candidates"]
            if Path(row["diagnostics_csv"]).name == path.name
        )
        points.append(
            {
                "label": f"TopK {candidate['top_k']}",
                "mean_fve": candidate["mean_validation_fve"],
                "mean_l0": candidate["mean_validation_l0"],
                "selected": candidate["config"] == selected["config"],
            }
        )
    original = None
    if args.original_diagnostics is not None and args.original_diagnostics.exists():
        original = aggregate_diagnostics(args.original_diagnostics)

    with args.screen.open("r", encoding="utf-8") as handle:
        screen = json.load(handle)
    if screen.get("status") != "complete":
        raise ValueError(f"Units feature screen is incomplete: {screen.get('status')!r}")
    discovery = sorted(
        screen["discovery"]["feature_results"],
        key=lambda row: int(row["discovery_rank"]),
    )[:20]
    confirmation_panels = [
        row
        for row in screen["confirmation"]["panels"]
        if row["kind"]
        in {"discovery_ranked_prefix", "full_graph_comparator", "reverse_rank_control"}
    ]
    primary_name = screen["confirmation"]["primary_result"]["panel"]
    primary = next(row for row in confirmation_panels if row["name"] == primary_name)

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.5), constrained_layout=True)

    axis = axes[0, 0]
    for point in points:
        axis.scatter(
            point["mean_l0"],
            point["mean_fve"],
            marker="*" if point["selected"] else "o",
            s=125 if point["selected"] else 58,
            color="#B22234" if point["selected"] else "#2166AC",
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        axis.annotate(
            point["label"] + (" selected" if point["selected"] else ""),
            (point["mean_l0"], point["mean_fve"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    if original is not None:
        axis.scatter(
            original["mean_l0"],
            original["mean_fve"],
            marker="s",
            s=58,
            color="#666666",
            edgecolor="white",
            linewidth=0.7,
        )
        axis.annotate(
            "Original ReLU",
            (original["mean_l0"], original["mean_fve"]),
            xytext=(5, -12),
            textcoords="offset points",
        )
    axis.axhline(
        selection["selection_rule"]["minimum_mean_validation_fve"],
        color="#777777",
        linestyle="--",
        linewidth=0.8,
    )
    axis.set_xscale("log")
    axis.set_xlabel("Mean active latents (log scale)")
    axis.set_ylabel("Mean validation FVE")
    axis.set_title("(a) Reconstruction-sparsity trade-off")
    axis.grid(color="#DDDDDD", linewidth=0.6)

    axis = axes[0, 1]
    shown = list(reversed(discovery))
    labels = [f"L{row['layer']} F{row['feature']}" for row in shown]
    values = [row["summary"]["mean_force_minus_mass_difference"] for row in shown]
    colours = [LAYER_COLOURS[int(row["layer"])] for row in shown]
    y = np.arange(len(shown))
    axis.barh(y, values, color=colours, edgecolor="white", linewidth=0.4)
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Discovery force-minus-mass gap delta")
    axis.set_title("(b) Discovery feature ranking")
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.6)

    axis = axes[1, 0]
    panel_labels = []
    means = []
    lower = []
    upper = []
    for panel in confirmation_panels:
        summary = panel["summary"]
        value = float(summary["mean_force_minus_mass_difference"])
        low, high = summary["bootstrap_95_ci_mean_force_minus_mass_difference"]
        panel_labels.append(
            panel["name"].replace("_primary", " (primary)").replace("_", " ")
        )
        means.append(value)
        lower.append(value - float(low))
        upper.append(float(high) - value)
    x = np.arange(len(confirmation_panels))
    axis.errorbar(
        x,
        means,
        yerr=np.asarray([lower, upper]),
        fmt="none",
        ecolor="#777777",
        elinewidth=1.0,
        capsize=3,
        zorder=1,
    )
    axis.scatter(
        x,
        means,
        c=["#B22234" if row["name"] == primary_name else "#2166AC" for row in confirmation_panels],
        s=42,
        edgecolor="white",
        linewidth=0.6,
        zorder=2,
    )
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_xticks(x, panel_labels, rotation=38, ha="right")
    axis.set_ylabel("Confirmation force-minus-mass gap delta")
    axis.set_title("(c) Frozen-panel confirmation")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)

    axis = axes[1, 1]
    effects = primary["case_effects"]
    for row in effects:
        axis.plot(
            [0, 1],
            [row["force_source_delta"], row["mass_source_delta"]],
            color="#AAAAAA",
            alpha=0.65,
            linewidth=0.8,
        )
        axis.scatter(0, row["force_source_delta"], color="#B22234", s=18, alpha=0.8)
        axis.scatter(1, row["mass_source_delta"], color="#2166AC", s=18, alpha=0.8)
    summary = primary["summary"]
    axis.scatter(
        [0, 1],
        [summary["mean_force_source_delta"], summary["mean_mass_source_delta"]],
        marker="D",
        s=64,
        color="#222222",
        label="Mean",
        zorder=3,
    )
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_xticks([0, 1], ["Force source", "Mass source control"])
    axis.set_ylabel("Change in newtons-vs-joules logit gap")
    axis.set_title("(d) Primary top-10 panel")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False)

    figure.suptitle(
        f"Units TopK SAE selection and causal validation (selected k={selected['top_k']})",
        fontsize=12,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_dir / f"{args.stem}.pdf", bbox_inches="tight")
    figure.savefig(args.output_dir / f"{args.stem}.png", dpi=230, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.output_dir / f'{args.stem}.pdf'}")


if __name__ == "__main__":
    main()

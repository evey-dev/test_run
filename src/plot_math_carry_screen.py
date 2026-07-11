"""Plot the discovery/confirmation mathematics carry feature screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.data_utils import get_repo_root, resolve_path


LAYER_COLOURS = {
    4: "#4C78A8",
    8: "#72B7B2",
    12: "#54A24B",
    16: "#ECA82C",
    20: "#F58518",
    24: "#E45756",
    28: "#B279A2",
}


def main_panels(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    panels = payload["confirmation"]["panels"]
    return [
        panel
        for panel in panels
        if panel["kind"] in {"discovery_ranked_prefix", "full_graph_comparator", "reverse_rank_control"}
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the carry feature screen")
    parser.add_argument(
        "--input",
        default="outputs/topk_math_followup/math_topk256_carry_feature_screen.json",
    )
    parser.add_argument("--output-dir", default="outputs/topk_math_followup/figures")
    args = parser.parse_args()

    repo_root = get_repo_root()
    input_path = resolve_path(args.input, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != "complete":
        raise ValueError(f"Feature screen is not complete: status={payload.get('status')!r}")

    feature_results = sorted(
        payload["discovery"]["feature_results"],
        key=lambda row: int(row["discovery_rank"]),
    )[:20]
    panels = main_panels(payload)
    primary_name = payload["confirmation"]["primary_result"]["panel"]
    primary = next(panel for panel in panels if panel["name"] == primary_name)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.titlesize": 14,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), gridspec_kw={"width_ratios": [1.1, 1.2, 1.0]})

    # (a) Discovery ordering.
    axis = axes[0]
    labels = [f"L{row['layer']} F{row['feature']}" for row in reversed(feature_results)]
    values = [row["summary"]["mean_paired_difference"] for row in reversed(feature_results)]
    colours = [LAYER_COLOURS[int(row["layer"])] for row in reversed(feature_results)]
    y = np.arange(len(labels))
    axis.barh(y, values, color=colours, edgecolor="white", linewidth=0.4)
    axis.axvline(0, color="#555555", linewidth=0.9)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Discovery carry-minus-control gap delta")
    axis.set_title("(a) Discovery feature ranking")
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    axis.set_axisbelow(True)

    # (b) Frozen confirmation panels.
    axis = axes[1]
    panel_labels = []
    means = []
    low_errors = []
    high_errors = []
    for panel in panels:
        label = panel["name"].replace("_primary", " (primary)").replace("_", " ")
        panel_labels.append(label)
        summary = panel["summary"]
        mean = float(summary["mean_paired_difference"])
        low, high = summary["bootstrap_95_ci_mean_paired_difference"]
        means.append(mean)
        low_errors.append(mean - float(low))
        high_errors.append(float(high) - mean)
    x = np.arange(len(panels))
    point_colours = ["#B22234" if panel["name"] == primary_name else "#4C78A8" for panel in panels]
    axis.errorbar(
        x,
        means,
        yerr=np.asarray([low_errors, high_errors]),
        fmt="none",
        ecolor="#777777",
        elinewidth=1.1,
        capsize=3,
        zorder=1,
    )
    axis.scatter(x, means, c=point_colours, s=42, edgecolor="white", linewidth=0.6, zorder=2)
    axis.axhline(0, color="#555555", linewidth=0.9)
    axis.set_xticks(x, panel_labels, rotation=40, ha="right")
    axis.set_ylabel("Confirmation carry-minus-control gap delta")
    axis.set_title("(b) Frozen-panel confirmation")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    axis.set_axisbelow(True)

    # (c) Primary paired cases.
    axis = axes[2]
    effects = primary["case_effects"]
    for row in effects:
        axis.plot(
            [0, 1],
            [row["target_delta"], row["control_delta"]],
            color="#A9A9A9",
            alpha=0.65,
            linewidth=0.9,
        )
        axis.scatter(0, row["target_delta"], color="#B22234", s=18, alpha=0.8, zorder=2)
        axis.scatter(1, row["control_delta"], color="#2F6F9F", s=18, alpha=0.8, zorder=2)
    summary = primary["summary"]
    axis.scatter(
        [0, 1],
        [summary["mean_target_delta"], summary["mean_no_carry_control_delta"]],
        marker="D",
        s=70,
        color="#222222",
        label="Mean",
        zorder=3,
    )
    axis.axhline(0, color="#555555", linewidth=0.9)
    axis.set_xticks([0, 1], ["Carry target", "No-carry control"])
    axis.set_ylabel("Correct-vs-alternative logit-gap delta")
    axis.set_title("(c) Primary top-10 panel")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="best")

    figure.suptitle("Final-position TopK SAE carry-feature discovery and confirmation")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    stem = output_dir / "fig_math_carry_feature_screen"
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {stem.with_suffix('.png')}")
    print(f"Saved {stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

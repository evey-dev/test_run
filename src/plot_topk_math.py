"""Plot the TopK SAE trade-off and matched carry-specificity experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_diagnostics(path: Path, label: str) -> Dict[str, Any]:
    rows = read_csv(path)
    return {
        "label": label,
        "mean_fve": mean(float(row["validation_fraction_variance_explained"]) for row in rows),
        "mean_l0": mean(float(row["validation_mean_l0"]) for row in rows),
    }


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TopK mathematics SAE comparison")
    parser.add_argument("--candidate-diagnostics", nargs="+", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--new-heldout", type=Path, required=True)
    parser.add_argument("--original-diagnostics", type=Path, default=None)
    parser.add_argument("--original-heldout", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/report_figures_topk"))
    parser.add_argument("--stem", default="fig_math_topk_tradeoff_specificity")
    args = parser.parse_args()

    with args.selection.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    selected = selection.get("selected")
    if selected is None:
        raise ValueError("Selection JSON contains no eligible candidate")

    candidate_by_path = {
        str(Path(candidate["diagnostics_csv"]).resolve()): candidate
        for candidate in selection["candidates"]
    }
    points = []
    for path in args.candidate_diagnostics:
        resolved = str(path.resolve())
        candidate = candidate_by_path.get(resolved)
        if candidate is None:
            candidate = next(
                item
                for item in selection["candidates"]
                if Path(item["diagnostics_csv"]).name == path.name
            )
        points.append(
            {
                "label": f"TopK {candidate['top_k']}",
                "top_k": candidate["top_k"],
                "mean_fve": candidate["mean_validation_fve"],
                "mean_l0": candidate["mean_validation_l0"],
                "selected": candidate["config"] == selected["config"],
            }
        )

    original_point = None
    if args.original_diagnostics is not None and args.original_diagnostics.exists():
        original_point = aggregate_diagnostics(args.original_diagnostics, "Original ReLU")

    with args.new_heldout.open("r", encoding="utf-8") as handle:
        heldout = json.load(handle)
    rows = [
        row
        for row in heldout["math"]["cases"]
        if row.get("eligible") and "specificity_control" in row
    ]
    target_deltas = [
        row["conditions"]["sparse_inhibition"]["gap"]
        - row["conditions"]["clean"]["gap"]
        for row in rows
    ]
    control_deltas = [row["specificity_control"]["gap_delta"] for row in rows]
    if not rows:
        raise ValueError("No baseline-qualified matched specificity-control rows were found")

    original_sparse_mean = None
    if args.original_heldout is not None and args.original_heldout.exists():
        with args.original_heldout.open("r", encoding="utf-8") as handle:
            original = json.load(handle)
        original_sparse_mean = original["math"]["summary"]["conditions"][
            "sparse_inhibition"
        ]["mean_gap_delta"]

    configure_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), constrained_layout=True)

    tradeoff = axes[0]
    for point in points:
        tradeoff.scatter(
            point["mean_l0"],
            point["mean_fve"],
            marker="*" if point["selected"] else "o",
            s=125 if point["selected"] else 58,
            color="#b2182b" if point["selected"] else "#2166ac",
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        tradeoff.annotate(
            point["label"] + (" selected" if point["selected"] else ""),
            (point["mean_l0"], point["mean_fve"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    if original_point is not None:
        tradeoff.scatter(
            original_point["mean_l0"],
            original_point["mean_fve"],
            marker="s",
            s=58,
            color="#666666",
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        tradeoff.annotate(
            original_point["label"],
            (original_point["mean_l0"], original_point["mean_fve"]),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=8,
        )
    threshold = selection["selection_rule"]["minimum_mean_validation_fve"]
    tradeoff.axhline(threshold, color="#777777", linestyle="--", linewidth=0.8)
    tradeoff.set_xscale("log")
    tradeoff.set_xlabel("Mean active latents (log scale)")
    tradeoff.set_ylabel("Mean validation FVE")
    tradeoff.set_title("(a) Reconstruction-sparsity trade-off")
    tradeoff.grid(color="#dddddd", linewidth=0.6)

    specificity = axes[1]
    for target, control in zip(target_deltas, control_deltas):
        specificity.plot([0, 1], [target, control], color="#bbbbbb", linewidth=0.8, zorder=1)
    specificity.scatter(
        [0] * len(target_deltas), target_deltas, color="#b2182b", alpha=0.72, s=28, zorder=2
    )
    specificity.scatter(
        [1] * len(control_deltas), control_deltas, color="#2166ac", alpha=0.72, s=28, zorder=2
    )
    specificity.scatter(
        [0, 1],
        [mean(target_deltas), mean(control_deltas)],
        marker="D",
        color="#222222",
        s=42,
        zorder=3,
        label="Mean",
    )
    if original_sparse_mean is not None:
        specificity.scatter(
            [-0.12],
            [original_sparse_mean],
            marker="x",
            color="#666666",
            s=48,
            linewidth=1.4,
            label="Original target mean",
            zorder=3,
        )
    specificity.axhline(0.0, color="#555555", linewidth=0.8)
    specificity.set_xticks([0, 1], ["Carry target", "No-carry control"])
    specificity.set_ylabel("Change in correct-vs-alternative logit gap")
    specificity.set_title(f"(b) Matched specificity control (TopK {selected['top_k']})")
    specificity.grid(axis="y", color="#dddddd", linewidth=0.6)
    specificity.legend(frameon=False, loc="best")

    fig.suptitle("Mathematics TopK SAE selection and causal validation", fontsize=11)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / f"{args.stem}.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / f"{args.stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output_dir / f'{args.stem}.pdf'}")


if __name__ == "__main__":
    main()

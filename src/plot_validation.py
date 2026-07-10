"""Create report-ready figures from the final validation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


DOMAIN_LABELS = {"math": "Addition", "units": "SI units", "capitals": "Capitals"}
DOMAIN_COLOURS = {"math": "#2166ac", "units": "#d6604d", "capitals": "#1b7837"}


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv_rows(paths: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                parsed: Dict[str, Any] = dict(row)
                for key, value in list(parsed.items()):
                    try:
                        parsed[key] = float(value)
                    except (TypeError, ValueError):
                        pass
                rows.append(parsed)
    return rows


def save_figure(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    print(f"Saved {output_dir / f'{stem}.pdf'}")
    print(f"Saved {output_dir / f'{stem}.png'}")


def plot_sae_diagnostics(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("validation_fraction_variance_explained", "Validation FVE", None),
        ("validation_mean_l0", "Mean active latents", None),
        ("combined_dead_feature_fraction", "Dead-feature fraction", (0.0, 1.0)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.1), constrained_layout=True)
    labels = sorted({str(row["label"]) for row in rows})
    for axis, (metric, title, ylim) in zip(axes, metrics):
        for label in labels:
            domain_rows = sorted(
                (row for row in rows if row["label"] == label), key=lambda row: row["layer"]
            )
            axis.plot(
                [row["layer"] for row in domain_rows],
                [row[metric] for row in domain_rows],
                marker="o",
                linewidth=1.7,
                markersize=4,
                label=DOMAIN_LABELS.get(label, label),
                color=DOMAIN_COLOURS.get(label),
            )
        axis.set_title(title)
        axis.set_xlabel("Transformer layer")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        if ylim is not None:
            axis.set_ylim(*ylim)
    axes[0].set_ylabel("Metric value")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("SAE fidelity and sparsity on the original validation splits", fontsize=11)
    save_figure(fig, output_dir, "fig_sae_diagnostics")
    plt.close(fig)


def eligible_deltas(domain_payload: Dict[str, Any]) -> Dict[str, np.ndarray]:
    rows = [row for row in domain_payload.get("cases", []) if row.get("eligible")]
    conditions = sorted({name for row in rows for name in row["conditions"] if name != "clean"})
    return {
        condition: np.asarray(
            [row["conditions"][condition]["gap"] - row["conditions"]["clean"]["gap"] for row in rows],
            dtype=float,
        )
        for condition in conditions
    }


def readable_condition(name: str) -> str:
    return {
        "sparse_inhibition": "Sparse inhibition",
        "sparse_feature_swap": "Sparse swap",
        "full_latent_swap": "Full latent",
        "raw_mlp_swap": "Raw MLP",
    }.get(name, name.replace("_", " ").title())


def plot_heldout(payload: Dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    available = [domain for domain in ("math", "units") if domain in payload]
    if not available:
        print("No held-out domains were present; skipping held-out figure")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(4.8 * len(available), 3.5), squeeze=False)
    rng = np.random.default_rng(787)
    for axis, domain in zip(axes[0], available):
        deltas = eligible_deltas(payload[domain])
        condition_names = list(deltas)
        for index, condition in enumerate(condition_names):
            values = deltas[condition]
            if values.size == 0:
                continue
            jitter = rng.normal(0.0, 0.045, size=values.size)
            colour = ["#4393c3", "#f4a582", "#5aae61"][index % 3]
            axis.scatter(
                np.full(values.size, index) + jitter,
                values,
                s=22,
                alpha=0.65,
                color=colour,
                edgecolor="white",
                linewidth=0.4,
                zorder=2,
            )
            axis.plot(index, values.mean(), marker="D", color="#222222", markersize=5, zorder=3)
        axis.axhline(0.0, color="#555555", linewidth=0.8)
        axis.set_xticks(range(len(condition_names)), [readable_condition(name) for name in condition_names])
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        eligible = payload[domain]["summary"]["eligible_cases"]
        total = payload[domain]["summary"]["total_cases"]
        axis.set_title(f"{DOMAIN_LABELS[domain]} ({eligible}/{total} qualified)")
        if domain == "math":
            axis.set_ylabel(r"Change in correct-minus-dropped-carry logit gap")
            axis.text(0.02, 0.02, "Predicted direction: below zero", transform=axis.transAxes, fontsize=8)
        else:
            axis.set_ylabel(r"Change in force-minus-energy prefix logit gap")
            axis.text(0.02, 0.02, "Predicted direction: above zero", transform=axis.transAxes, fontsize=8)
    fig.suptitle("Graph-held-out intervention effects", fontsize=11)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_heldout_generalisation")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot final SAE diagnostics and held-out tests")
    parser.add_argument("--diagnostics", nargs="+", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/report_figures"))
    args = parser.parse_args()

    configure_matplotlib()
    diagnostic_rows = read_csv_rows(args.diagnostics)
    plot_sae_diagnostics(diagnostic_rows, args.output_dir)
    with args.heldout.open("r", encoding="utf-8") as handle:
        heldout_payload = json.load(handle)
    plot_heldout(heldout_payload, args.output_dir)


if __name__ == "__main__":
    main()

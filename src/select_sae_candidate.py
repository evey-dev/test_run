"""Select an SAE candidate using reconstruction and sparsity metrics only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from src.data_utils import get_repo_root, resolve_path


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarise_candidate(config: str, diagnostics_path: Path) -> Dict[str, Any]:
    rows = read_rows(diagnostics_path)
    if not rows:
        raise ValueError(f"No diagnostic rows found in {diagnostics_path}")

    activation_types = {row["activation_type"] for row in rows}
    top_k_values = {int(float(row["top_k"])) for row in rows if row.get("top_k")}
    if activation_types != {"topk"} or len(top_k_values) != 1:
        raise ValueError(
            f"Expected one TopK candidate in {diagnostics_path}, found "
            f"activation_type={activation_types}, top_k={top_k_values}"
        )

    fve = [float(row["validation_fraction_variance_explained"]) for row in rows]
    l0 = [float(row["validation_mean_l0"]) for row in rows]
    dead = [float(row["combined_dead_feature_fraction"]) for row in rows]
    decoder_medians = [float(row["decoder_norm_median"]) for row in rows]
    top_k = next(iter(top_k_values))
    if max(l0) > top_k + 1e-3:
        raise ValueError(
            f"Observed L0 exceeds k for {diagnostics_path}: max L0={max(l0):.3f}, k={top_k}"
        )

    return {
        "config": config,
        "diagnostics_csv": str(diagnostics_path),
        "activation_type": "topk",
        "top_k": top_k,
        "layers": [int(float(row["layer"])) for row in rows],
        "mean_validation_fve": mean(fve),
        "minimum_layer_validation_fve": min(fve),
        "mean_validation_l0": mean(l0),
        "maximum_layer_validation_l0": max(l0),
        "mean_dead_feature_fraction": mean(dead),
        "mean_decoder_norm_median": mean(decoder_medians),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Choose the sparsest SAE candidate that meets fixed FVE thresholds"
    )
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--diagnostics", nargs="+", required=True, type=Path)
    parser.add_argument("--minimum-mean-fve", type=float, default=0.90)
    parser.add_argument("--minimum-layer-fve", type=float, default=0.85)
    parser.add_argument("--maximum-mean-dead-fraction", type=float, default=0.80)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if len(args.configs) != len(args.diagnostics):
        parser.error("--configs and --diagnostics must contain the same number of paths")

    repo_root = get_repo_root()
    candidates = [
        summarise_candidate(config, resolve_path(path, repo_root))
        for config, path in zip(args.configs, args.diagnostics)
    ]
    for candidate in candidates:
        candidate["meets_threshold"] = bool(
            candidate["mean_validation_fve"] >= args.minimum_mean_fve
            and candidate["minimum_layer_validation_fve"] >= args.minimum_layer_fve
            and candidate["mean_dead_feature_fraction"] <= args.maximum_mean_dead_fraction
        )

    eligible = [candidate for candidate in candidates if candidate["meets_threshold"]]
    selected = min(
        eligible,
        key=lambda candidate: (
            candidate["mean_validation_l0"],
            -candidate["minimum_layer_validation_fve"],
        ),
        default=None,
    )
    payload = {
        "selection_rule": {
            "minimum_mean_validation_fve": args.minimum_mean_fve,
            "minimum_layer_validation_fve": args.minimum_layer_fve,
            "maximum_mean_dead_feature_fraction": args.maximum_mean_dead_fraction,
            "choice": "smallest mean validation L0 among candidates meeting all fidelity and collapse thresholds",
            "intervention_results_used_for_selection": False,
        },
        "candidates": sorted(candidates, key=lambda candidate: candidate["top_k"]),
        "selected": selected,
    }

    output_path = resolve_path(args.output, repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("\nTopK candidate comparison")
    print("k       mean_FVE    min_FVE    mean_L0    dead_fraction    eligible")
    for candidate in payload["candidates"]:
        print(
            f"{candidate['top_k']:<7d} "
            f"{candidate['mean_validation_fve']:>8.4f} "
            f"{candidate['minimum_layer_validation_fve']:>10.4f} "
            f"{candidate['mean_validation_l0']:>10.1f} "
            f"{candidate['mean_dead_feature_fraction']:>16.4f} "
            f"{str(candidate['meets_threshold']):>11s}"
        )
    if selected is None:
        print("\nNo candidate met the predeclared FVE thresholds. Do not build a graph yet.")
    else:
        print(f"\nSelected without intervention results: {selected['config']} (k={selected['top_k']})")
    print(f"Saved selection record to {output_path}")


if __name__ == "__main__":
    main()

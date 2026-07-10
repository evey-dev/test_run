"""Measure reconstruction fidelity and sparsity of trained SAE checkpoints.

The training script records the optimisation objective, but that scalar does not
separate reconstruction quality from latent magnitude and does not report whether
the learned code is actually sparse. This module evaluates each saved checkpoint
on the original train/validation split and writes report-ready JSON and CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F

from src.config_utils import load_yaml_config
from src.data_utils import get_repo_root, load_activation_splits, resolve_path
from src.train import SparseAutoencoder


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def batched_indices(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def load_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_split(
    model: SparseAutoencoder,
    activations: np.ndarray,
    indices: np.ndarray,
    train_mean_norm: torch.Tensor,
    scaling_factor: float,
    batch_size: int,
    activity_epsilon: float,
    device: str,
) -> Dict[str, Any]:
    sum_squared_error = 0.0
    sum_squared_total = 0.0
    sum_cosine = 0.0
    sum_relative_l2 = 0.0
    sum_l0 = 0.0
    sum_l1 = 0.0
    sample_count = 0
    element_count = 0
    active_features = torch.zeros(model.encoder.out_features, dtype=torch.bool)
    l0_values: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch_idx in batched_indices(indices, batch_size):
            batch_np = np.asarray(activations[batch_idx], dtype=np.float32)
            x = torch.from_numpy(batch_np).to(device=device)
            x_norm = x / scaling_factor
            x_hat_norm, z = model(x_norm)
            residual = x_norm - x_hat_norm

            sum_squared_error += float(residual.square().sum().item())
            sum_squared_total += float((x_norm - train_mean_norm).square().sum().item())
            sum_cosine += float(F.cosine_similarity(x_norm, x_hat_norm, dim=-1).sum().item())
            relative_l2 = residual.norm(dim=-1) / x_norm.norm(dim=-1).clamp_min(1e-12)
            sum_relative_l2 += float(relative_l2.sum().item())

            active = z > activity_epsilon
            l0 = active.sum(dim=-1)
            sum_l0 += float(l0.sum().item())
            sum_l1 += float(z.abs().sum().item())
            active_features |= active.any(dim=0).cpu()
            l0_values.append(l0.cpu().numpy())

            sample_count += int(x.shape[0])
            element_count += int(x.numel())

    l0_array = np.concatenate(l0_values) if l0_values else np.asarray([], dtype=float)
    fvu = sum_squared_error / max(sum_squared_total, 1e-12)
    latent_dim = model.encoder.out_features
    return {
        "samples": sample_count,
        "normalized_mse": sum_squared_error / max(element_count, 1),
        "fraction_variance_unexplained": fvu,
        "fraction_variance_explained": 1.0 - fvu,
        "mean_cosine_similarity": sum_cosine / max(sample_count, 1),
        "mean_relative_l2_error": sum_relative_l2 / max(sample_count, 1),
        "mean_l0": sum_l0 / max(sample_count, 1),
        "median_l0": float(np.median(l0_array)) if l0_array.size else 0.0,
        "p95_l0": float(np.percentile(l0_array, 95)) if l0_array.size else 0.0,
        "mean_active_fraction": sum_l0 / max(sample_count * latent_dim, 1),
        "mean_latent_l1": sum_l1 / max(sample_count * latent_dim, 1),
        "active_feature_count": int(active_features.sum().item()),
        "dead_feature_count": int((~active_features).sum().item()),
        "dead_feature_fraction": float((~active_features).float().mean().item()),
        "active_feature_mask": active_features,
    }


def evaluate_layer(
    layer: int,
    cfg: Dict[str, Any],
    label: str,
    batch_size: int,
    activity_epsilon: float,
    device: str,
) -> Dict[str, Any]:
    repo_root = get_repo_root()
    data_dir = resolve_path(cfg["data_dir"], repo_root)
    sae_dir = resolve_path(cfg["output_dir"], repo_root)
    activation_path = data_dir / f"activations_layer{layer}.npy"
    checkpoint_path = sae_dir / f"sae_layer{layer}.pt"
    metadata_path = sae_dir / f"sae_layer{layer}_metadata.json"

    if not activation_path.exists():
        raise FileNotFoundError(f"Activation file not found: {activation_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint_path}")

    metadata = load_metadata(metadata_path)
    scaling_factor = float(metadata.get("activation_scaling_factor", 1.0))
    activations = np.load(activation_path, mmap_mode="r")
    splits = load_activation_splits(data_dir)
    train_idx = np.asarray(splits[layer]["train"], dtype=int)
    val_idx = np.asarray(splits[layer]["val"], dtype=int)

    hidden_size = int(cfg.get("hidden_size", activations.shape[-1]))
    latent_dim = int(cfg.get("latent_dim", 8192))
    model = SparseAutoencoder(hidden_size, latent_dim)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device=device, dtype=torch.float32)

    train_mean = np.asarray(activations[train_idx], dtype=np.float32).mean(axis=0)
    train_mean_norm = torch.from_numpy(train_mean / scaling_factor).to(device=device)

    train_metrics = evaluate_split(
        model,
        activations,
        train_idx,
        train_mean_norm,
        scaling_factor,
        batch_size,
        activity_epsilon,
        device,
    )
    val_metrics = evaluate_split(
        model,
        activations,
        val_idx,
        train_mean_norm,
        scaling_factor,
        batch_size,
        activity_epsilon,
        device,
    )

    union_active = train_metrics.pop("active_feature_mask") | val_metrics.pop("active_feature_mask")
    decoder_norms = model.decoder.weight.detach().float().norm(dim=0).cpu().numpy()
    encoder_norms = model.encoder.weight.detach().float().norm(dim=1).cpu().numpy()
    history = metadata.get("history", [])
    best_epoch = None
    if history:
        best_epoch = int(min(history, key=lambda row: float(row["val_loss"]))["epoch"])

    result = {
        "label": label,
        "layer": layer,
        "hidden_size": hidden_size,
        "latent_dim": latent_dim,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "activity_epsilon": activity_epsilon,
        "activation_scaling_factor": scaling_factor,
        "best_epoch": best_epoch,
        "saved_best_val_loss": metadata.get("best_val_loss"),
        "train": train_metrics,
        "validation": val_metrics,
        "combined_dead_feature_count": int((~union_active).sum().item()),
        "combined_dead_feature_fraction": float((~union_active).float().mean().item()),
        "decoder_norm_p05": float(np.percentile(decoder_norms, 5)),
        "decoder_norm_median": float(np.median(decoder_norms)),
        "decoder_norm_p95": float(np.percentile(decoder_norms, 95)),
        "decoder_norm_max": float(decoder_norms.max()),
        "encoder_norm_p05": float(np.percentile(encoder_norms, 5)),
        "encoder_norm_median": float(np.median(encoder_norms)),
        "encoder_norm_p95": float(np.percentile(encoder_norms, 95)),
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def flatten_for_csv(result: Dict[str, Any]) -> Dict[str, Any]:
    row = {key: value for key, value in result.items() if not isinstance(value, dict)}
    for split_name in ("train", "validation"):
        for key, value in result[split_name].items():
            row[f"{split_name}_{key}"] = value
    return row


def print_summary(results: List[Dict[str, Any]]) -> None:
    print("\nSAE diagnostic summary (validation split)")
    print("domain      layer      FVE   rel_L2   mean_L0   dead_all")
    for result in results:
        val = result["validation"]
        print(
            f"{result['label']:<11} {result['layer']:>5} "
            f"{val['fraction_variance_explained']:>8.3f} "
            f"{val['mean_relative_l2_error']:>8.3f} "
            f"{val['mean_l0']:>10.1f} "
            f"{result['combined_dead_feature_fraction']:>10.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained SAE reconstruction and sparsity")
    parser.add_argument("--config", required=True, help="Behaviour-specific SAE YAML config")
    parser.add_argument("--label", required=True, help="Short domain label written to outputs")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--activity-epsilon", type=float, default=1e-6)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    repo_root = get_repo_root()
    cfg = load_yaml_config(repo_root / args.config)
    layers = args.layers or [int(layer) for layer in cfg.get("layers", [])]
    device = resolve_device(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    results = []
    for layer in layers:
        print(f"Evaluating {args.label} SAE at layer {layer} on {device}...")
        results.append(
            evaluate_layer(
                layer,
                cfg,
                args.label,
                args.batch_size,
                args.activity_epsilon,
                device,
            )
        )

    output_json = resolve_path(args.output_json, repo_root)
    output_csv = resolve_path(args.output_csv, repo_root)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": {
            "config": args.config,
            "activity_epsilon": args.activity_epsilon,
            "fve_definition": "1 - SSE(reconstruction) / SSE(validation activation - training mean)",
            "dead_feature_definition": "latent never exceeds activity_epsilon on train or validation examples",
        },
        "layers": results,
    }
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    rows = [flatten_for_csv(result) for result in results]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print_summary(results)
    print(f"\nSaved JSON: {output_json}")
    print(f"Saved CSV:  {output_csv}")


if __name__ == "__main__":
    main()

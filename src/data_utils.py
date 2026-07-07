"""Utility helpers for locating the repository root and loading activation data files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


def get_repo_root() -> Path:
    """Return the repository root based on this file location."""
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a path relative to the repo root or a provided base directory."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    base = Path(base_dir) if base_dir is not None else get_repo_root()
    if not base.is_absolute():
        base = get_repo_root() / base
    return (base / candidate).resolve()


def verify_activation_file(path: str | Path) -> Dict[str, Any]:
    """Load a NumPy activation file and return basic metadata."""
    arr = np.load(path)
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def find_activation_data_dir(data_dir: str | Path | None = None) -> Path:
    """Resolve the activation bundle directory, preferring generated hidden artifacts when present."""
    repo_root = get_repo_root()
    candidates: List[Path] = []

    if data_dir is not None:
        candidates.append(resolve_path(data_dir, repo_root))

    candidates.extend(
        [
            resolve_path("mechanistic_data", repo_root),
            resolve_path("mechanistic_data", repo_root),
        ]
    )

    for candidate in candidates:
        split_path = candidate / "train_val_indices_per_layer.npy"
        activation_glob = any(candidate.glob("activations_layer*.npy"))
        if split_path.exists() or activation_glob:
            return candidate

    return resolve_path(data_dir or "mechanistic_data", repo_root)


def load_activation_tensor(layer: int | str, data_dir: str | Path | None = None) -> np.ndarray:
    """Load a single layer activation tensor from disk."""
    data_dir_path = find_activation_data_dir(data_dir)
    activation_path = data_dir_path / f"activations_layer{layer}.npy"
    if not activation_path.exists():
        raise FileNotFoundError(f"Activation file not found: {activation_path}")
    return np.load(activation_path)


def load_activation_splits(data_dir: str | Path | None = None) -> Dict[int, Dict[str, np.ndarray]]:
    """Load the train/validation split indices for each layer."""
    data_dir_path = find_activation_data_dir(data_dir)
    split_path = data_dir_path / "train_val_indices_per_layer.npy"
    if not split_path.exists():
        raise FileNotFoundError(f"Split index file not found: {split_path}")

    raw = np.load(split_path, allow_pickle=True).item()
    processed: Dict[int, Dict[str, np.ndarray]] = {}
    for layer, payload in raw.items():
        processed[int(layer)] = {
            "train": np.asarray(payload["train"], dtype=int),
            "val": np.asarray(payload["val"], dtype=int),
        }
    return processed


def save_activation_metadata(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    """Persist activation metadata as a CSV file."""
    pd.DataFrame(rows).to_csv(path, index=False)

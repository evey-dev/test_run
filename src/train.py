"""Train sparse autoencoders on precomputed activation tensors.

The script loads activation tensors and train/validation split indices, trains
one or more SAEs for the requested layers, and saves weights, latents, and
metadata to the output directory.

Optional CLI arguments:
- --config: path to the SAE YAML configuration file
- --layers: override the layers to train
- --epochs: override the number of training epochs
- --batch-size: override the training batch size
- --latent-dim: override the latent dimension
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config_utils import ensure_output_dir, load_yaml_config
from src.data_utils import get_repo_root, load_activation_splits, load_activation_tensor, resolve_path


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_latent: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(d_in, d_latent)
        self.decoder = nn.Linear(d_latent, d_in, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(d_in))
        self.register_buffer("scaling_factor", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Subtract decoder bias before encoding
        x_centered = x - self.decoder_bias
        z = torch.relu(self.encoder(x_centered))
        # Add decoder bias after decoding
        x_hat = self.decoder(z) + self.decoder_bias
        return x_hat, z


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loss_fn(x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor, l1_lambda: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = ((x - x_hat) ** 2).mean()
    l1 = z.abs().mean()
    return mse + l1_lambda * l1, mse, l1


def resolve_device(device_setting: str | None) -> str:
    if device_setting in {None, "", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_setting


def train_sae(layer: int, x_train: torch.Tensor, x_val: torch.Tensor, cfg: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    print(f"\n=== Training SAE for layer {layer} ===")
    device = resolve_device(cfg.get("device"))
    hidden_size = int(cfg.get("hidden_size", 2560))
    latent_dim = int(cfg.get("latent_dim", 8192))
    batch_size = int(cfg.get("batch_size", 64))
    epochs = int(cfg.get("epochs", 50))
    lr = float(cfg.get("lr", 1e-3))
    l1_lambda = float(cfg.get("l1_lambda", 1e-3))

    # Activation Normalization: scale so mean L2 norm is sqrt(hidden_size)
    raw_mean_l2 = torch.norm(x_train, dim=-1).mean().item()
    if raw_mean_l2 == 0:
        raw_mean_l2 = 1.0
    scaling_factor = raw_mean_l2 / (hidden_size ** 0.5)
    print(f"Layer {layer} activation scaling factor (raw mean L2 / sqrt(d_in)): {scaling_factor:.4f}")

    x_train_norm = x_train / scaling_factor
    x_val_norm = x_val / scaling_factor

    model = SparseAutoencoder(hidden_size, latent_dim).to(device)
    model.scaling_factor.copy_(torch.tensor(scaling_factor))

    # Pre-bias decoder to mean of normalized training activations
    with torch.no_grad():
        model.decoder_bias.copy_(x_train_norm.mean(dim=0))

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(TensorDataset(x_train_norm), batch_size=batch_size, shuffle=True)
    x_val_norm = x_val_norm.to(device)

    best_state = None
    best_val_loss = float("inf")
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            x_hat, z = model(batch)
            loss, _, _ = loss_fn(batch, x_hat, z, l1_lambda)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            x_hat, z = model(x_val_norm)
            val_loss, val_mse, val_l1 = loss_fn(x_val_norm, x_hat, z, l1_lambda)

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": float(val_loss.item()),
                "val_mse": float(val_mse.item()),
                "val_l1": float(val_l1.item()),
            }
        )
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss.item():.4f} | "
            f"mse={val_mse.item():.4f} | "
            f"l1={val_l1.item():.4f}"
        )
        if float(val_loss.item()) < best_val_loss:
            best_val_loss = float(val_loss.item())
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    model.load_state_dict(best_state)
    return model, {
        "best_val_loss": best_val_loss,
        "history": history,
        "scaling_factor": scaling_factor
    }


def prepare_layer_data(layer: int, cfg: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    data_dir = resolve_path(cfg.get("data_dir", "mechanistic_data"))
    activations = load_activation_tensor(layer, data_dir=data_dir)
    splits = load_activation_splits(data_dir=data_dir)

    if layer not in splits:
        raise KeyError(f"Missing split indices for layer {layer}")

    x = torch.tensor(activations, dtype=torch.float32)
    idx_train = splits[layer]["train"]
    idx_val = splits[layer]["val"]
    return x[idx_train], x[idx_val]


def save_latents(model: nn.Module, x: torch.Tensor, layer: int, out_dir: Path, device: str) -> None:
    model.eval()
    with torch.no_grad():
        _, z = model(x.to(device))
        z_np = z.cpu().numpy()
    np.save(out_dir / f"latents_layer{layer}.npy", z_np)


def save_metadata(layer: int, cfg: Dict[str, Any], history: List[Dict[str, float]], best_val_loss: float, scaling_factor: float, out_dir: Path) -> None:
    metadata = {
        "layer": layer,
        "hidden_size": int(cfg.get("hidden_size", 2560)),
        "latent_dim": int(cfg.get("latent_dim", 8192)),
        "batch_size": int(cfg.get("batch_size", 64)),
        "epochs": int(cfg.get("epochs", 50)),
        "lr": float(cfg.get("lr", 1e-3)),
        "l1_lambda": float(cfg.get("l1_lambda", 1e-3)),
        "seed": int(cfg.get("seed", 787)),
        "best_val_loss": float(best_val_loss),
        "activation_scaling_factor": float(scaling_factor),
        "history": history,
    }
    with open(out_dir / f"sae_layer{layer}_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def _layer_artifacts(layer: int) -> List[str]:
    """File names written for a single layer's SAE (checkpoint, latents, metadata)."""
    return [f"sae_layer{layer}.pt", f"latents_layer{layer}.npy", f"sae_layer{layer}_metadata.json"]


def _copy_layer_to_drive(layer: int, out_dir: Path, drive_dir: Path) -> None:
    """Copy a finished layer's artifacts to a persistent (Drive) directory."""
    drive_dir.mkdir(parents=True, exist_ok=True)
    for name in _layer_artifacts(layer):
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, drive_dir / name)
    print(f"[drive] Backed up layer {layer} artifacts to {drive_dir}")


def _restore_layer_from_drive(layer: int, out_dir: Path, drive_dir: Path) -> bool:
    """If a layer is already trained in Drive, copy it back locally and report True.

    A layer counts as complete only when all of its artifacts are present in Drive.
    """
    names = _layer_artifacts(layer)
    if not all((drive_dir / name).exists() for name in names):
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(drive_dir / name, out_dir / name)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sparse autoencoders on precomputed activations")
    parser.add_argument("--config", default="configs/sae_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml", help="Path to the model config file if auto-generating activations")
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Override the layers to train")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--drive-dir", default=None, help="Persistent directory (e.g. a mounted Google Drive path) to back up each layer to as it finishes and resume from on restart")
    args = parser.parse_args()

    repo_root = get_repo_root()
    cfg = load_yaml_config(repo_root / args.config)
    if args.layers is not None:
        cfg["layers"] = args.layers
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.latent_dim is not None:
        cfg["latent_dim"] = args.latent_dim

    set_seed(int(cfg.get("seed", 787)))
    out_dir = ensure_output_dir(resolve_path(cfg.get("output_dir", "mechanistic_data/sae_checkpoints"), repo_root))
    device = resolve_device(cfg.get("device"))
    cfg["device"] = device

    # Auto-generate activations if they are missing
    data_dir = resolve_path(cfg.get("data_dir", "mechanistic_data"), repo_root)
    layers_to_check = cfg.get("layers", [8, 16, 24, 32])
    
    missing_any = False
    for layer in layers_to_check:
        act_file = data_dir / f"activations_layer{layer}.npy"
        if not act_file.exists():
            missing_any = True
            break
            
    if not (data_dir / "train_val_indices_per_layer.npy").exists():
        missing_any = True

    if missing_any:
        print("\n[Auto-generation] Activations or split indices are missing. Running activation capture on-the-fly...")
        from src.capture_activations import capture_activations
        capture_activations(
            model_config_path=args.model_config,
            output_dir=data_dir,
            layers=layers_to_check,
            seed=int(cfg.get("seed", 787))
        )
        print("[Auto-generation] Activations captured successfully!\n")

    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available; using CPU")

    drive_dir = Path(args.drive_dir).expanduser() if args.drive_dir else None

    for layer in cfg.get("layers", [8, 16, 24, 32]):
        layer = int(layer)

        # Resume: if this layer was already trained and backed up to Drive, restore and skip.
        if drive_dir is not None and _restore_layer_from_drive(layer, out_dir, drive_dir):
            print(f"[resume] Layer {layer} already complete in {drive_dir}; skipping training.")
            continue

        x_train, x_val = prepare_layer_data(layer, cfg)
        model, training_info = train_sae(layer, x_train, x_val, cfg)

        torch.save(model.state_dict(), out_dir / f"sae_layer{layer}.pt")
        save_metadata(layer, cfg, training_info["history"], training_info["best_val_loss"], training_info["scaling_factor"], out_dir)

        x_full = torch.cat([x_train, x_val], dim=0)
        x_full_norm = x_full / training_info["scaling_factor"]
        save_latents(model, x_full_norm, layer, out_dir, device)
        print(f"Saved SAE + latents for layer {layer}")

        # Persist this layer immediately so a later crash doesn't lose it.
        if drive_dir is not None:
            _copy_layer_to_drive(layer, out_dir, drive_dir)


if __name__ == "__main__":
    main()
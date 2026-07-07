"""Capture real MLP activations from the Qwen3-4B-Instruct model on prompt datasets."""

import argparse
import os
import shutil
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config_utils import ensure_output_dir, load_yaml_config
from src.data_utils import get_repo_root, resolve_path
from src.model_loader import load_model_and_tokenizer, set_seed
from src.prompts.prompt_utils import load_prompts, format_prompt

DEFAULT_LAYERS = (8, 16, 24, 32)
DEFAULT_BEHAVIOURS = ("capitals", "addition", "units")

def copy_model_checkpoint(source_dir: Path, target_dir: Path) -> None:
    """Save a current checkpoint of the model by copying its directory."""
    print(f"Saving checkpoint of model from {source_dir} to {target_dir}...")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    print("Model checkpoint saved successfully.")

def capture_activations(
    model_config_path: str | None = None,
    output_dir: str | Path | None = None,
    behaviours: Sequence[str] = DEFAULT_BEHAVIOURS,
    layers: Sequence[int] = DEFAULT_LAYERS,
    copy_model: bool = False,
    seed: int = 787,
) -> Path:
    set_seed(seed)
    repo_root = get_repo_root()
    out_path = resolve_path(output_dir or "mechanistic_data", repo_root)
    ensure_output_dir(out_path)

    # 1. Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer, model_cfg = load_model_and_tokenizer(model_config_path)
    
    # Optional checkpoint copy
    if copy_model:
        source_model_dir = Path(model_cfg["model_dir"])
        target_model_dir = out_path / "original_model"
        copy_model_checkpoint(source_model_dir, target_model_dir)

    # 2. Load and merge prompts for chosen behaviours
    prompt_rows: List[Dict[str, Any]] = []
    for behaviour in behaviours:
        print(f"Loading prompts for behaviour: {behaviour}")
        try:
            p_data = load_prompts(behaviour)
            for idx, p in enumerate(p_data["prompts"]):
                prompt_rows.append({
                    "id": p["id"],
                    "behaviour": behaviour,
                    "sentence": format_prompt(p),
                    "prompt_idx": idx,
                })
        except Exception as e:
            print(f"Warning: Failed to load prompts for {behaviour}: {e}")

    if not prompt_rows:
        raise ValueError("No prompts loaded! Make sure data files exist in the data/ directory.")

    print(f"Total prompts loaded: {len(prompt_rows)}")

    # 3. Setup hooks to capture activations
    # We will accumulate outputs for each layer
    activations_by_layer: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # output shape: [batch_size, seq_len, hidden_size]
            # Capture the last token activation
            # output is bfloat16, convert to float32 on CPU
            last_token_act = output[:, -1, :].detach().cpu().to(torch.float32)
            activations_by_layer[layer_idx].append(last_token_act)
        return hook_fn

    hooks = []
    for layer in layers:
        if layer < 0 or layer >= len(model.model.layers):
            raise ValueError(f"Invalid layer index {layer} for model with {len(model.model.layers)} layers")
        # In Qwen3, the MLP block is at model.layers[i].mlp
        mlp_module = model.model.layers[layer].mlp
        h = mlp_module.register_forward_hook(make_hook(layer))
        hooks.append(h)

    # 4. Run forward pass on all prompts (one by one to avoid padding issues)
    print("Running forward passes to capture activations...")
    metadata_rows: List[Dict[str, Any]] = []
    
    with torch.no_grad():
        for i, row in enumerate(tqdm(prompt_rows)):
            text = row["sentence"]
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]
            last_token_id = input_ids[0, -1].item()
            last_token = tokenizer.decode([last_token_id])

            # Run forward pass (triggers hooks)
            model(**inputs)

            # Record metadata
            for layer in layers:
                metadata_rows.append({
                    "prompt_id": i,
                    "layer": int(layer),
                    "token_position": seq_len - 1,
                    "token": last_token,
                    "prompt": text,
                    "seed": seed,
                    "model": model_cfg["model_dir"],
                    "hidden_size": int(model_cfg.get("hidden_size", 2560)),
                    "num_layers": len(model.model.layers),
                    "behaviour": row["behaviour"],
                    "prompt_id_within_behaviour": row["id"],
                })

    # Remove hooks
    for h in hooks:
        h.remove()

    # 5. Process and save activations per layer
    print("Saving activation tensors to disk...")
    for layer in layers:
        # Concatenate list of [1, hidden_size] tensors along dim 0
        layer_tensors = torch.cat(activations_by_layer[layer], dim=0) # [num_prompts, hidden_size]
        layer_np = layer_tensors.numpy()
        np.save(out_path / f"activations_layer{layer}.npy", layer_np)
        print(f"Saved activations for layer {layer}: {layer_np.shape} to {out_path / f'activations_layer{layer}.npy'}")

    # 6. Save split indices
    print("Generating train/val split indices...")
    num_prompts = len(prompt_rows)
    indices = np.arange(num_prompts)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    split_ratio = 0.8
    split_idx = int(round(num_prompts * split_ratio))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    np.save(out_path / "train_indices.npy", train_idx)
    np.save(out_path / "val_indices.npy", val_idx)
    
    splits_dict = {}
    for layer in layers:
        splits_dict[int(layer)] = {
            "train": train_idx.astype(int),
            "val": val_idx.astype(int),
        }
    np.save(out_path / "train_val_indices_per_layer.npy", splits_dict, allow_pickle=True)

    # 7. Save metadata CSV
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(out_path / "activation_metadata.csv", index=False)
    print(f"Saved metadata to {out_path / 'activation_metadata.csv'}")

    return out_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Capture real MLP activations from Qwen3")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--output-dir", default="mechanistic_data", help="Output directory for activations")
    parser.add_argument("--behaviours", nargs="+", default=list(DEFAULT_BEHAVIOURS), help="Behaviours to capture")
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS), help="Layers to capture")
    parser.add_argument("--copy-model", action="store_true", help="Copy the model directory into the output folder")
    parser.add_argument("--seed", type=int, default=787)
    args = parser.parse_args()

    capture_activations(
        model_config_path=args.model_config,
        output_dir=args.output_dir,
        behaviours=args.behaviours,
        layers=args.layers,
        copy_model=args.copy_model,
        seed=args.seed,
    )
    print("Activation capture completed successfully!")

if __name__ == "__main__":
    main()

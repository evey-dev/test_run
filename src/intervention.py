"""Validate computational circuits using inhibition and activation swap-in interventions."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

from src.config_utils import ensure_output_dir, load_yaml_config
from src.data_utils import get_repo_root, resolve_path
from src.model_loader import load_model_and_tokenizer
from src.train import SparseAutoencoder


def load_sae_models(
    layers: List[int],
    sae_dir: Path,
    hidden_size: int,
    latent_dim: int,
    device: str,
    dtype: torch.dtype
) -> Dict[int, Tuple[SparseAutoencoder, float]]:
    """Load SAE models and their scaling factors for the specified layers."""
    saes = {}
    for layer in layers:
        checkpoint_path = sae_dir / f"sae_layer{layer}.pt"
        metadata_path = sae_dir / f"sae_layer{layer}_metadata.json"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint_path}")
        
        scaling_factor = 1.0
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
                scaling_factor = float(meta.get("activation_scaling_factor", 1.0))
        
        sae = SparseAutoencoder(hidden_size, latent_dim)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        sae.load_state_dict(state_dict)
        sae.to(device=device, dtype=dtype)
        sae.eval()
        
        saes[layer] = (sae, scaling_factor)
        
    return saes


def get_baseline_predictions(model, tokenizer, prompt: str) -> Tuple[torch.Tensor, int, str]:
    """Get the model's output logits, top-1 predicted token ID, and token string."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    top_id = torch.argmax(logits).item()
    top_tok = tokenizer.decode([top_id])
    return logits, top_id, top_tok


def run_inhibition_intervention(
    model,
    tokenizer,
    prompt: str,
    layers: List[int],
    saes: Dict[int, Tuple[SparseAutoencoder, float]],
    inhibited_features: Dict[int, List[int]],
    target_tokens: List[str] | None = None
) -> Dict[str, Any]:
    """Zero-out specific feature activations and measure changes in model output."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    hooks = []
    
    # Reconstruction and inhibition hooks
    def make_inhibition_hook(layer_idx, sae_model, scaling_factor, f_list):
        def hook_fn(module, input_t, output_t):
            last_token_act = output_t[:, -1, :]
            last_token_norm = last_token_act / scaling_factor
            
            x_centered = last_token_norm - sae_model.decoder_bias
            z = torch.relu(sae_model.encoder(x_centered))
            
            # Perform inhibition on specified feature indices
            for f_idx in f_list:
                if f_idx < z.shape[1]:
                    z[:, f_idx] = 0.0
                    
            x_hat_norm = sae_model.decoder(z) + sae_model.decoder_bias
            x_hat = x_hat_norm * scaling_factor
            
            new_output = output_t.clone()
            new_output[:, -1, :] = x_hat
            return new_output
        return hook_fn

    # Register hooks for all layers, only adding inhibition where specified
    for layer in layers:
        sae_model, scaling_factor = saes[layer]
        f_list = inhibited_features.get(layer, [])
        h = model.model.layers[layer].mlp.register_forward_hook(
            make_inhibition_hook(layer, sae_model, scaling_factor, f_list)
        )
        hooks.append(h)

    print(f"Running forward pass with inhibition: {inhibited_features}...")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    
    # Remove hooks
    for h in hooks:
        h.remove()
        
    probs = torch.softmax(logits, dim=-1)
    
    # Get top predicted token under intervention
    intervened_top_id = torch.argmax(logits).item()
    intervened_top_tok = tokenizer.decode([intervened_top_id])
    
    # Resolve target tokens if provided
    targets_info = {}
    if target_tokens:
        for t in target_tokens:
            t_id = tokenizer.convert_tokens_to_ids(t)
            if t_id == tokenizer.unk_token_id:
                ids = tokenizer.encode(t, add_special_tokens=False)
                if ids:
                    t_id = ids[0]
            if t_id != tokenizer.unk_token_id:
                targets_info[t] = {
                    "logit": float(logits[t_id].item()),
                    "prob": float(probs[t_id].item())
                }
            
    return {
        "top_token": intervened_top_tok,
        "top_logit": float(logits[intervened_top_id].item()),
        "top_prob": float(probs[intervened_top_id].item()),
        "targets": targets_info
    }


def run_swap_in_intervention(
    model,
    tokenizer,
    source_prompt: str,
    target_prompt: str,
    layers: List[int],
    saes: Dict[int, Tuple[SparseAutoencoder, float]],
    swap_features: Dict[int, List[int]] | None = None,
    target_tokens: List[str] | None = None
) -> Dict[str, Any]:
    """Capture activations from a source prompt and swap them into the target prompt run."""
    # 1. Capture source prompt latents
    source_z = {}
    hooks = []
    
    def make_capture_hook(layer_idx, sae_model, scaling_factor):
        def hook_fn(module, input_t, output_t):
            last_token_act = output_t[:, -1, :]
            last_token_norm = last_token_act / scaling_factor
            x_centered = last_token_norm - sae_model.decoder_bias
            z = torch.relu(sae_model.encoder(x_centered))
            source_z[layer_idx] = z.detach()
            return output_t
        return hook_fn
        
    for layer in layers:
        sae_model, scaling_factor = saes[layer]
        h = model.model.layers[layer].mlp.register_forward_hook(
            make_capture_hook(layer, sae_model, scaling_factor)
        )
        hooks.append(h)
        
    source_inputs = tokenizer(source_prompt, return_tensors="pt").to(model.device)
    print(f"Capturing source activations on prompt: '{source_prompt}'...")
    with torch.no_grad():
        model(**source_inputs)
        
    for h in hooks:
        h.remove()
        
    # 2. Run target prompt while swapping in source activations
    swap_hooks = []
    
    def make_swap_hook(layer_idx, sae_model, scaling_factor, src_z, f_list):
        def hook_fn(module, input_t, output_t):
            last_token_act = output_t[:, -1, :]
            last_token_norm = last_token_act / scaling_factor
            
            x_centered = last_token_norm - sae_model.decoder_bias
            z = torch.relu(sae_model.encoder(x_centered))
            
            # Align src_z device/dtype
            device_src_z = src_z.to(device=z.device, dtype=z.dtype)
            
            if f_list is None:
                # Swap entire latent layer
                z = device_src_z.clone()
            else:
                # Swap only specific feature indices
                for f_idx in f_list:
                    if f_idx < z.shape[1]:
                        z[:, f_idx] = device_src_z[:, f_idx]
                        
            x_hat_norm = sae_model.decoder(z) + sae_model.decoder_bias
            x_hat = x_hat_norm * scaling_factor
            
            new_output = output_t.clone()
            new_output[:, -1, :] = x_hat
            return new_output
        return hook_fn
        
    for layer in layers:
        sae_model, scaling_factor = saes[layer]
        src_z = source_z[layer]
        f_list = swap_features.get(layer) if swap_features else None
        h = model.model.layers[layer].mlp.register_forward_hook(
            make_swap_hook(layer, sae_model, scaling_factor, src_z, f_list)
        )
        swap_hooks.append(h)
        
    target_inputs = tokenizer(target_prompt, return_tensors="pt").to(model.device)
    print(f"Running target pass with swap-in on prompt: '{target_prompt}'...")
    with torch.no_grad():
        outputs = model(**target_inputs)
        
    for h in swap_hooks:
        h.remove()
        
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    
    intervened_top_id = torch.argmax(logits).item()
    intervened_top_tok = tokenizer.decode([intervened_top_id])
    
    targets_info = {}
    if target_tokens:
        for t in target_tokens:
            t_id = tokenizer.convert_tokens_to_ids(t)
            if t_id == tokenizer.unk_token_id:
                ids = tokenizer.encode(t, add_special_tokens=False)
                if ids:
                    t_id = ids[0]
            if t_id != tokenizer.unk_token_id:
                targets_info[t] = {
                    "logit": float(logits[t_id].item()),
                    "prob": float(probs[t_id].item())
                }
            
    return {
        "top_token": intervened_top_tok,
        "top_logit": float(logits[intervened_top_id].item()),
        "top_prob": float(probs[intervened_top_id].item()),
        "targets": targets_info
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SAE circuits with interventions")
    parser.add_argument("--mode", required=True, choices=["inhibit", "swap"], help="Intervention type: inhibit or swap")
    parser.add_argument("--prompt", required=True, help="Prompt string (or target prompt for swap mode)")
    parser.add_argument("--source-prompt", default=None, help="Source prompt for swap mode")
    parser.add_argument("--features", default=None, help="JSON string specifying features. Format: '{\"8\": [12, 34], \"16\": [56]}'")
    parser.add_argument("--target-token", default=None, help="Target next token to track")
    parser.add_argument("--layers", nargs="+", type=int, default=[8, 16, 24, 32], help="Layers involved in the experiment")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--sae-config", default="configs/sae_config.yaml")
    parser.add_argument("--output", default="outputs/intervention_results.json", help="Path to save output results")
    args = parser.parse_args()

    repo_root = get_repo_root()
    sae_cfg = load_yaml_config(repo_root / args.sae_config)
    sae_dir = resolve_path(sae_cfg.get("output_dir", "mechanistic_data/sae_checkpoints"), repo_root)
    hidden_size = int(sae_cfg.get("hidden_size", 2560))
    latent_dim = int(sae_cfg.get("latent_dim", 8192))
    
    print("Loading model...")
    model, tokenizer, model_cfg = load_model_and_tokenizer(repo_root / args.model_config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saes = load_sae_models(args.layers, sae_dir, hidden_size, latent_dim, device, model.dtype)

    # Parse features dictionary if provided
    features_dict = {}
    if args.features:
        try:
            features_dict = {int(k): (list(v) if v is not None else None) for k, v in raw_features.items()}
        except Exception as e:
            raise ValueError(f"Could not parse features JSON '{args.features}': {e}")

    # 1. Run baseline on prompt (target prompt for swap mode)
    base_logits, base_id, base_tok = get_baseline_predictions(model, tokenizer, args.prompt)
    base_probs = torch.softmax(base_logits, dim=-1)
    
    # Automatically track the top 3 predictions of the target prompt baseline
    auto_tokens_set = set()
    _, top_base_ids = torch.topk(base_probs, 3)
    for idx in top_base_ids:
        auto_tokens_set.add(tokenizer.decode([idx.item()]))
        
    # Automatically track the top 3 predictions of the source prompt baseline (if in swap mode)
    src_probs = None
    src_logits = None
    if args.mode == "swap" and args.source_prompt:
        src_logits, src_id, src_tok = get_baseline_predictions(model, tokenizer, args.source_prompt)
        src_probs = torch.softmax(src_logits, dim=-1)
        _, top_src_ids = torch.topk(src_probs, 3)
        for idx in top_src_ids:
            auto_tokens_set.add(tokenizer.decode([idx.item()]))
            
    # Add user target tokens (including stripped/space-prefixed variants for safety)
    if args.target_token:
        user_tokens = [t for t in args.target_token.split(",") if t.strip()]
        for ut in user_tokens:
            ut_clean = ut.strip().strip("'\"")
            auto_tokens_set.add(ut_clean)
            auto_tokens_set.add(" " + ut_clean)
            auto_tokens_set.add(ut)
            
    # Resolve to unique list of decoded token strings
    target_tokens = []
    seen_ids = set()
    for tok in auto_tokens_set:
        t_id = tokenizer.convert_tokens_to_ids(tok)
        if t_id == tokenizer.unk_token_id:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            if ids:
                t_id = ids[0]
        if t_id != tokenizer.unk_token_id and t_id not in seen_ids:
            seen_ids.add(t_id)
            target_tokens.append(tokenizer.decode([t_id]))
            
    # Sort for deterministic printing
    target_tokens.sort(key=lambda t: t.strip().lower())

    print(f"\n[1/3] Clean Model Baseline:")
    print(f"  - Top prediction: '{base_tok}' (prob: {base_probs[base_id].item():.4f}, logit: {base_logits[base_id].item():.2f})")
    
    results = {
        "experiment_mode": args.mode,
        "prompt": args.prompt,
        "clean_baseline": {
            "top_token": base_tok,
            "top_logit": float(base_logits[base_id].item()),
            "top_prob": float(base_probs[base_id].item()),
            "targets": {}
        }
    }
    
    for t in target_tokens:
        t_id = tokenizer.convert_tokens_to_ids(t)
        if t_id == tokenizer.unk_token_id:
            ids = tokenizer.encode(t, add_special_tokens=False)
            if ids:
                t_id = ids[0]
        if t_id != tokenizer.unk_token_id:
            results["clean_baseline"]["targets"][t] = {
                "logit": float(base_logits[t_id].item()),
                "prob": float(base_probs[t_id].item())
            }
            t_rep = t.replace("\n", "\\n").replace("\r", "\\r")
            print(f"  - Target '{t_rep}': prob: {base_probs[t_id].item():.4f}, logit: {base_logits[t_id].item():.2f}")

    # 2. Run Intervention/Reconstruction
    if args.mode == "inhibit":
        # Run SAE reconstruction-only baseline (no features inhibited)
        print("\n[2/3] Running SAE Reconstruction-only Baseline (no features inhibited)...")
        recon_res = run_inhibition_intervention(
            model, tokenizer, args.prompt, args.layers, saes, {}, target_tokens
        )
        results["reconstruction_baseline"] = recon_res
        print(f"  - Top prediction: '{recon_res['top_token']}' (prob: {recon_res['top_prob']:.4f}, logit: {recon_res['top_logit']:.2f})")
        for t, info in recon_res.get("targets", {}).items():
            t_rep = t.replace("\n", "\\n").replace("\r", "\\r")
            print(f"  - Target '{t_rep}': prob: {info['prob']:.4f}, logit: {info['logit']:.2f}")
            
        # Run actual inhibition intervention
        print(f"\n[3/3] Running Inhibition Intervention (inhibited features: {features_dict})...")
        inter_res = run_inhibition_intervention(
            model, tokenizer, args.prompt, args.layers, saes, features_dict, target_tokens
        )
        results["intervention"] = inter_res
        print(f"  - Top prediction: '{inter_res['top_token']}' (prob: {inter_res['top_prob']:.4f}, logit: {inter_res['top_logit']:.2f})")
        for t, info in inter_res.get("targets", {}).items():
            t_rep = t.replace("\n", "\\n").replace("\r", "\\r")
            print(f"  - Target '{t_rep}': prob: {info['prob']:.4f}, logit: {info['logit']:.2f}")
            
    elif args.mode == "swap":
        if not args.source_prompt:
            raise ValueError("source-prompt is required in swap mode")
        
        # Log source baseline predictions
        print(f"\nSource baseline prediction on '{args.source_prompt}':")
        results["source_prompt"] = args.source_prompt
        results["source_baseline"] = {
            "top_token": src_tok if 'src_tok' in locals() else tokenizer.decode([src_id]),
            "top_logit": float(src_logits[src_id].item()),
            "top_prob": float(src_probs[src_id].item()),
            "targets": {}
        }
        
        for t in target_tokens:
            t_id = tokenizer.convert_tokens_to_ids(t)
            if t_id == tokenizer.unk_token_id:
                ids = tokenizer.encode(t, add_special_tokens=False)
                if ids:
                    t_id = ids[0]
            if t_id != tokenizer.unk_token_id:
                results["source_baseline"]["targets"][t] = {
                    "logit": float(src_logits[t_id].item()),
                    "prob": float(src_probs[t_id].item())
                }
                t_rep = t.replace("\n", "\\n").replace("\r", "\\r")
                print(f"  - Target '{t_rep}': prob: {src_probs[t_id].item():.4f}, logit: {src_logits[t_id].item():.2f}")
        
        print(f"\nRunning Swap-In Intervention (swapping features {features_dict} from source to target)...")
        inter_res = run_swap_in_intervention(
            model, tokenizer, args.source_prompt, args.prompt, args.layers, saes, 
            features_dict if features_dict else None, target_tokens
        )
        results["intervention"] = inter_res
        print(f"  - Top prediction: '{inter_res['top_token']}' (prob: {inter_res['top_prob']:.4f}, logit: {inter_res['top_logit']:.2f})")
        for t, info in inter_res.get("targets", {}).items():
            t_rep = t.replace("\n", "\\n").replace("\r", "\\r")
            print(f"  - Target '{t_rep}': prob: {info['prob']:.4f}, logit: {info['logit']:.2f}")

    # Save results
    output_path = resolve_path(args.output, repo_root)
    ensure_output_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved intervention results to {output_path}")


if __name__ == "__main__":
    main()

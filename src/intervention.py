"""Validate computational circuits using inhibition and activation swap-in interventions."""

import argparse
import json
import os
import re
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
        activation_type = "relu"
        top_k = None
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
                scaling_factor = float(meta.get("activation_scaling_factor", 1.0))
                activation_type = str(meta.get("activation_type", "relu"))
                top_k = meta.get("top_k")
        
        sae = SparseAutoencoder(
            hidden_size,
            latent_dim,
            activation_type=activation_type,
            top_k=top_k,
        )
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


def resolve_positions(position_spec: str, seq_len: int) -> List[int]:
    """Resolve a position spec into concrete token indices."""
    spec = position_spec.strip().lower()
    if spec in {"last", "final"}:
        return [seq_len - 1]
    if spec == "all":
        return list(range(seq_len))

    positions: List[int] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0:
            idx = seq_len + idx
        if idx < 0 or idx >= seq_len:
            raise ValueError(f"Position {raw_part!r} resolved to {idx}, outside sequence length {seq_len}")
        if idx not in positions:
            positions.append(idx)
    if not positions:
        raise ValueError(f"Could not parse --positions value: {position_spec!r}")
    return positions


def token_position_records(tokenizer, input_ids: List[int], positions: List[int]) -> List[Dict[str, Any]]:
    """Return readable token metadata for selected prompt positions."""
    records = []
    for idx in positions:
        tok_id = int(input_ids[idx])
        tok = tokenizer.decode([tok_id]).replace("\n", "\\n").replace("\r", "\\r")
        records.append({"position": int(idx), "token_id": tok_id, "token": tok})
    return records


def print_position_selection(tokenizer, input_ids: List[int], positions: List[int], label: str = "Editing") -> None:
    """Print the exact prompt tokens that an intervention will edit."""
    records = token_position_records(tokenizer, input_ids, positions)
    print(f"{label} token positions:")
    for rec in records:
        print(f"  {rec['position']:>2}: id={rec['token_id']:<8} token={rec['token']!r}")


def print_token_positions(tokenizer, prompt: str) -> None:
    """Print prompt token indices for choosing --positions values."""
    encoded = tokenizer(prompt, return_tensors="pt")
    ids = encoded["input_ids"][0].tolist()
    print_position_selection(tokenizer, ids, list(range(len(ids))), label="Prompt")
    print(f"Use --positions last, --positions all, or comma-separated indices such as --positions 4,5,9")


def run_inhibition_intervention(
    model,
    tokenizer,
    prompt: str,
    layers: List[int],
    saes: Dict[int, Tuple[SparseAutoencoder, float]],
    inhibited_features: Dict[int, List[int]],
    target_tokens: List[str] | None = None,
    edit_strength: float = 1.0,
    position_spec: str = "last"
) -> Dict[str, Any]:
    """Zero-out specific feature activations and measure changes in model output.

    Uses an *error-preserving* edit: rather than replacing the MLP output with the
    (lossy) SAE reconstruction, we subtract only the decoder contribution of the
    ablated features from the model's true activation. Because the SAE decoder has
    no bias, the reconstruction error and the decoder bias cancel exactly, so:
      - ablating zero features reproduces the clean run bit-for-bit, and
      - ablating a feature subtracts precisely that feature's decoder direction.
    This isolates the causal effect of the feature instead of burying it under the
    reconstruction error of every hooked layer.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    position_idxs = resolve_positions(position_spec, inputs["input_ids"].shape[1])
    input_ids = inputs["input_ids"][0].detach().cpu().tolist()
    position_records = token_position_records(tokenizer, input_ids, position_idxs)
    hooks = []
    # Record the pre-ablation activation of each targeted feature for diagnostics.
    captured_feature_acts: Dict[int, Dict[int, Dict[str, Any]]] = {}

    def make_inhibition_hook(layer_idx, sae_model, scaling_factor, f_list):
        def hook_fn(module, input_t, output_t):
            selected_act = output_t[:, position_idxs, :]
            flat_act = selected_act.reshape(-1, selected_act.shape[-1])
            flat_norm = flat_act / scaling_factor

            z = sae_model.encode(flat_norm)

            if not f_list:
                return output_t  # No edit at this layer: leave the true activation untouched.

            # Record activations of the targeted features, then build the ablated code.
            layer_acts = {}
            z_ablated = z.clone()
            for f_idx in f_list:
                if 0 <= f_idx < z.shape[1]:
                    vals = z[:, f_idx].detach().float().cpu().tolist()
                    by_position = []
                    for pos_rec, act in zip(position_records, vals):
                        by_position.append({
                            "position": pos_rec["position"],
                            "token": pos_rec["token"],
                            "activation": float(act),
                        })
                    layer_acts[int(f_idx)] = {
                        "max": float(max(vals)) if vals else 0.0,
                        "mean": float(np.mean(vals)) if vals else 0.0,
                        "by_position": by_position,
                    }
                    z_ablated[:, f_idx] = 0.0
            captured_feature_acts[layer_idx] = layer_acts

            # Error-preserving edit: add only the *change* in reconstruction (decoder is bias-free).
            delta_norm = sae_model.decoder(z_ablated - z) * edit_strength
            new_selected = selected_act + (delta_norm * scaling_factor).reshape_as(selected_act)

            new_output = output_t.clone()
            new_output[:, position_idxs, :] = new_selected
            return new_output
        return hook_fn

    # Register hooks only where features are actually being inhibited.
    for layer in layers:
        sae_model, scaling_factor = saes[layer]
        f_list = inhibited_features.get(layer, [])
        h = model.model.layers[layer].mlp.register_forward_hook(
            make_inhibition_hook(layer, sae_model, scaling_factor, f_list)
        )
        hooks.append(h)

    print(f"Running forward pass with inhibition at positions {position_idxs}: {inhibited_features}...")
    print_position_selection(tokenizer, input_ids, position_idxs)
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
        "targets": targets_info,
        "feature_activations": captured_feature_acts,
        "positions": position_idxs,
        "position_tokens": position_records,
    }


def run_swap_in_intervention(
    model,
    tokenizer,
    source_prompt: str,
    target_prompt: str,
    layers: List[int],
    saes: Dict[int, Tuple[SparseAutoencoder, float]],
    swap_features: Dict[int, List[int]] | None = None,
    target_tokens: List[str] | None = None,
    edit_strength: float = 1.0,
    raw_mlp_swap: bool = False,
    position_spec: str = "last"
) -> Dict[str, Any]:
    """Capture activations from a source prompt and swap them into the target prompt run.

    The swap can target any set of token positions (``position_spec``), not just the
    final token. For a minimal-pair activation patch (e.g. a carry vs no-carry prompt
    that differ in a single token), the source and target must tokenize to the same
    length so positions line up; this is validated below.
    """
    source_inputs = tokenizer(source_prompt, return_tensors="pt").to(model.device)
    target_inputs = tokenizer(target_prompt, return_tensors="pt").to(model.device)
    src_len = source_inputs["input_ids"].shape[1]
    tgt_len = target_inputs["input_ids"].shape[1]

    # Positions are resolved against the target sequence; the same indices are read
    # from the captured source activations, so alignment requires equal lengths for
    # any non-final position spec.
    position_idxs = resolve_positions(position_spec, tgt_len)
    if position_spec.strip().lower() not in {"last", "final"} and src_len != tgt_len:
        raise ValueError(
            f"Position-aligned swap needs equal-length prompts, but source has {src_len} "
            f"tokens and target has {tgt_len}. Use --positions last, or pick a minimal "
            f"pair that tokenizes to the same length."
        )
    tgt_input_ids = target_inputs["input_ids"][0].detach().cpu().tolist()
    position_records = token_position_records(tokenizer, tgt_input_ids, position_idxs)

    # 1. Capture source prompt latents (and raw MLP outputs) at the selected positions.
    source_z = {}
    source_raw_mlp = {}
    hooks = []

    def make_capture_hook(layer_idx, sae_model, scaling_factor):
        def hook_fn(module, input_t, output_t):
            sel_act = output_t[:, position_idxs, :]
            source_raw_mlp[layer_idx] = sel_act.detach()
            if raw_mlp_swap:
                return output_t
            flat = sel_act.reshape(-1, sel_act.shape[-1]) / scaling_factor
            z = sae_model.encode(flat)
            source_z[layer_idx] = z.detach()
            return output_t
        return hook_fn

    for layer in layers:
        sae_model, scaling_factor = saes[layer]
        h = model.model.layers[layer].mlp.register_forward_hook(
            make_capture_hook(layer, sae_model, scaling_factor)
        )
        hooks.append(h)

    print(f"Capturing source activations on prompt: '{source_prompt}' at positions {position_idxs}...")
    with torch.no_grad():
        model(**source_inputs)

    for h in hooks:
        h.remove()

    # 2. Run target prompt while swapping in source activations at the same positions.
    swap_hooks = []

    def make_swap_hook(layer_idx, sae_model, scaling_factor, src_z, f_list):
        def hook_fn(module, input_t, output_t):
            sel_act = output_t[:, position_idxs, :]
            flat = sel_act.reshape(-1, sel_act.shape[-1])
            flat_norm = flat / scaling_factor

            z = sae_model.encode(flat_norm)

            # Align src_z device/dtype
            device_src_z = src_z.to(device=z.device, dtype=z.dtype)

            z_swapped = z.clone()
            if f_list is None:
                # Swap the entire latent code from the source run.
                z_swapped = device_src_z.clone()
            elif len(f_list) == 0:
                return output_t  # Layer not targeted: leave the true activation untouched.
            else:
                # Swap only the specified feature indices.
                for f_idx in f_list:
                    if 0 <= f_idx < z.shape[1]:
                        z_swapped[:, f_idx] = device_src_z[:, f_idx]

            # Error-preserving edit: apply only the change in reconstruction to the
            # target's true activation. With f_list=None this equals the full source
            # reconstruction plus the target's own reconstruction error.
            delta_norm = sae_model.decoder(z_swapped - z) * edit_strength
            new_sel = sel_act + (delta_norm * scaling_factor).reshape_as(sel_act)

            new_output = output_t.clone()
            new_output[:, position_idxs, :] = new_sel
            return new_output
        return hook_fn

    for layer in layers:
        sae_model, scaling_factor = saes[layer]
        src_z = source_z.get(layer)
        src_raw = source_raw_mlp[layer]
        # swap_features is None => swap entire latent code at every layer.
        # swap_features is a dict => only touch layers it names; others are a no-op ([]).
        f_list = None if swap_features is None else swap_features.get(layer, [])
        if raw_mlp_swap:
            # In a raw-MLP layer scan, non-selected layers are represented by an
            # empty feature list. Honour that no-op marker instead of patching
            # every requested layer and mislabelling the result as single-layer.
            if f_list == []:
                continue

            def make_raw_swap_hook(layer_idx, raw_sel):
                def hook_fn(module, input_t, output_t):
                    new_output = output_t.clone()
                    new_output[:, position_idxs, :] = raw_sel.to(device=output_t.device, dtype=output_t.dtype)
                    return new_output
                return hook_fn
            h = model.model.layers[layer].mlp.register_forward_hook(
                make_raw_swap_hook(layer, src_raw)
            )
        else:
            h = model.model.layers[layer].mlp.register_forward_hook(
                make_swap_hook(layer, sae_model, scaling_factor, src_z, f_list)
            )
        swap_hooks.append(h)

    print(f"Running target pass with swap-in on prompt: '{target_prompt}' at positions {position_idxs}...")
    print_position_selection(tokenizer, tgt_input_ids, position_idxs, label="Swapping into")
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
        "targets": targets_info,
        "positions": position_idxs,
        "position_tokens": position_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SAE circuits with interventions")
    parser.add_argument("--mode", required=True, choices=["inhibit", "swap"], help="Intervention type: inhibit or swap")
    parser.add_argument("--prompt", required=True, help="Prompt string (or target prompt for swap mode)")
    parser.add_argument("--source-prompt", default=None, help="Source prompt for swap mode")
    parser.add_argument("--features", default=None, help="JSON string specifying features. Format: '{\"8\": [12, 34], \"16\": [56]}'")
    parser.add_argument("--target-token", default=None, help="Target next token to track")
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 8, 12, 16, 20, 24, 28, 32], help="Layers involved in the experiment")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--sae-config", default="configs/sae_config.yaml")
    parser.add_argument("--output", default="outputs/intervention_results.json", help="Path to save output results")
    parser.add_argument("--graph-json", default=None, help="Path to attribution graph JSON; extracts all features from nodes to use as ablation targets")
    parser.add_argument("--graph-feature-sign", choices=["all", "positive", "negative"], default="all", help="When using --graph-json, keep all features, only positive-attribution features, or only negative-attribution features")
    parser.add_argument("--edit-strength", type=float, default=1.0, help="Multiplier for SAE decoder-delta edits. 1.0 is a literal ablation/swap; >1 is a stress-test diagnostic")
    parser.add_argument("--raw-mlp-swap", action="store_true", help="Swap raw source MLP outputs instead of SAE latents at the positions selected by --positions (upper-bound diagnostic)")
    parser.add_argument("--scan", action="store_true", help="Progressive ablation scan: ablate top-10/25/50/100/ALL features by attribution magnitude")
    parser.add_argument("--full-knockout", action="store_true", help="Zero out an entire component at selected token positions for all hooked layers (diagnostic)")
    parser.add_argument("--knockout-component", choices=["mlp", "attn", "block"], default="mlp", help="Component to zero for --full-knockout. 'block' zeros the layer output hidden state at the last token.")
    parser.add_argument("--layer-scan", action="store_true", help="With --full-knockout, also run one knockout per layer to localize which layer matters most")
    parser.add_argument("--position-scan", action="store_true", help="With --full-knockout, also run one knockout per selected token position")
    parser.add_argument("--positions", default="last", help="Token positions to edit: 'last', 'all', or comma-separated indices such as '4,5,9'. Negative indices are allowed.")
    parser.add_argument("--print-tokens", action="store_true", help="Print prompt token positions and exit")
    args = parser.parse_args()

    repo_root = get_repo_root()
    sae_cfg = load_yaml_config(repo_root / args.sae_config)
    sae_dir = resolve_path(sae_cfg.get("output_dir", "mechanistic_data/sae_checkpoints"), repo_root)
    hidden_size = int(sae_cfg.get("hidden_size", 2560))
    latent_dim = int(sae_cfg.get("latent_dim", 8192))
    
    print("Loading model...")
    model, tokenizer, model_cfg = load_model_and_tokenizer(repo_root / args.model_config)

    if args.print_tokens:
        print_token_positions(tokenizer, args.prompt)
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saes = load_sae_models(args.layers, sae_dir, hidden_size, latent_dim, device, model.dtype)

    # Parse features dictionary if provided
    features_dict = {}
    if args.features:
        try:
            raw_features = json.loads(args.features)
            features_dict = {int(k): (list(v) if v is not None else None) for k, v in raw_features.items()}
        except Exception as e:
            raise ValueError(f"Could not parse features JSON '{args.features}': {e}")

    # Load graph JSON and extract features (overrides --features if both provided)
    graph_nodes = []  # Raw node dicts, kept for --scan sorting by attribution
    if args.graph_json:
        graph_path = Path(args.graph_json)
        if not graph_path.is_absolute():
            graph_path = repo_root / graph_path
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph JSON not found: {graph_path}")
        with open(graph_path, "r", encoding="utf-8") as fh:
            graph_data = json.load(fh)
        graph_nodes = graph_data.get("nodes", [])
        # Build features_dict from graph nodes whose id matches "layer_X_feature_Y".
        # Positive attribution means the feature supports the graphed target logit;
        # negative attribution means it suppresses that logit. For inhibition, mixing
        # both signs can cancel the effect, so --graph-feature-sign positive is often
        # the cleanest "remove support for this answer" test.
        graph_features_dict: Dict[int, List[int]] = {}
        skipped_by_sign = 0
        skipped_by_layer = 0
        for node in graph_nodes:
            node_id = node.get("id", "")
            m = re.match(r"layer_(\d+)_feature_(\d+)", node_id)
            if m:
                layer_idx = int(m.group(1))
                if layer_idx not in args.layers:
                    skipped_by_layer += 1
                    continue
                attribution = float(node.get("attribution", 0.0))
                if args.graph_feature_sign == "positive" and attribution <= 0:
                    skipped_by_sign += 1
                    continue
                if args.graph_feature_sign == "negative" and attribution >= 0:
                    skipped_by_sign += 1
                    continue
                feat_idx = int(m.group(2))
                graph_features_dict.setdefault(layer_idx, []).append(feat_idx)
        features_dict = graph_features_dict
        print(f"Loaded {len(graph_nodes)} nodes from graph JSON; extracted features across {len(features_dict)} layers "
              f"({sum(len(v) for v in features_dict.values())} total features)")
        if skipped_by_layer:
            print(f"  Layer filter: using layers {args.layers} (skipped {skipped_by_layer} graph feature nodes)")
        if args.graph_feature_sign != "all":
            print(f"  Graph feature sign filter: {args.graph_feature_sign} (skipped {skipped_by_sign} feature nodes)")

    # Validate --scan usage
    if args.scan and not args.graph_json:
        raise ValueError("--scan requires --graph-json to determine feature ordering by attribution")

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
            model, tokenizer, args.prompt, args.layers, saes, {}, target_tokens, position_spec=args.positions
        )
        results["reconstruction_baseline"] = recon_res
        print(f"  - Top prediction: '{recon_res['top_token']}' (prob: {recon_res['top_prob']:.4f}, logit: {recon_res['top_logit']:.2f})")
        for t, info in recon_res.get("targets", {}).items():
            t_rep = t.replace("\n", "\\n").replace("\r", "\\r")
            print(f"  - Target '{t_rep}': prob: {info['prob']:.4f}, logit: {info['logit']:.2f}")

        # --full-knockout: zero out an entire component at selected token positions.
        if args.full_knockout:
            def run_component_knockout(
                knockout_layers: List[int],
                show_positions: bool = False,
                position_spec_override: str | None = None
            ) -> Dict[str, Any]:
                inputs_ko = tokenizer(args.prompt, return_tensors="pt").to(model.device)
                position_spec = position_spec_override if position_spec_override is not None else args.positions
                position_idxs = resolve_positions(position_spec, inputs_ko["input_ids"].shape[1])
                input_ids = inputs_ko["input_ids"][0].detach().cpu().tolist()
                position_records = token_position_records(tokenizer, input_ids, position_idxs)
                if show_positions:
                    print_position_selection(
                        tokenizer,
                        input_ids,
                        position_idxs,
                        label=f"{args.knockout_component.upper()} knockout editing",
                    )
                ko_hooks = []
                component_activation_norms: Dict[int, List[Dict[str, Any]]] = {}

                def make_full_knockout_hook(layer_idx):
                    def hook_fn(module, input_t, output_t):
                        # MLP modules return a tensor. Attention/layer modules may return
                        # either a tensor or a tuple whose first item is hidden states.
                        if isinstance(output_t, tuple):
                            original_hidden = output_t[0]
                            selected = original_hidden[:, position_idxs, :].detach().float()
                            l2_norms = selected.norm(dim=-1)[0].cpu().tolist()
                            mean_abs = selected.abs().mean(dim=-1)[0].cpu().tolist()
                            component_activation_norms[layer_idx] = [
                                {
                                    "position": rec["position"],
                                    "token": rec["token"],
                                    "l2_norm": float(l2),
                                    "mean_abs": float(ma),
                                }
                                for rec, l2, ma in zip(position_records, l2_norms, mean_abs)
                            ]
                            hidden = original_hidden.clone()
                            hidden[:, position_idxs, :] = 0.0
                            return (hidden,) + output_t[1:]
                        selected = output_t[:, position_idxs, :].detach().float()
                        l2_norms = selected.norm(dim=-1)[0].cpu().tolist()
                        mean_abs = selected.abs().mean(dim=-1)[0].cpu().tolist()
                        component_activation_norms[layer_idx] = [
                            {
                                "position": rec["position"],
                                "token": rec["token"],
                                "l2_norm": float(l2),
                                "mean_abs": float(ma),
                            }
                            for rec, l2, ma in zip(position_records, l2_norms, mean_abs)
                        ]
                        new_output = output_t.clone()
                        new_output[:, position_idxs, :] = 0.0
                        return new_output
                    return hook_fn

                for layer in knockout_layers:
                    block = model.model.layers[layer]
                    if args.knockout_component == "mlp":
                        module = block.mlp
                    elif args.knockout_component == "attn":
                        module = block.self_attn
                    else:
                        module = block
                    h = module.register_forward_hook(make_full_knockout_hook(layer))
                    ko_hooks.append(h)

                with torch.no_grad():
                    ko_outputs = model(**inputs_ko)
                ko_logits = ko_outputs.logits[0, -1, :]

                for h in ko_hooks:
                    h.remove()

                ko_probs = torch.softmax(ko_logits, dim=-1)
                ko_top_id = torch.argmax(ko_logits).item()
                ko_top_tok = tokenizer.decode([ko_top_id])
                ko_result = {
                    "top_token": ko_top_tok,
                    "top_logit": float(ko_logits[ko_top_id].item()),
                    "top_prob": float(ko_probs[ko_top_id].item()),
                    "knockout_component": args.knockout_component,
                    "layers": knockout_layers,
                    "position_spec": position_spec,
                    "positions": position_idxs,
                    "position_tokens": position_records,
                    "component_activation_norms": component_activation_norms,
                    "targets": {}
                }
                for t in target_tokens:
                    t_id = tokenizer.convert_tokens_to_ids(t)
                    if t_id == tokenizer.unk_token_id:
                        ids = tokenizer.encode(t, add_special_tokens=False)
                        if ids:
                            t_id = ids[0]
                    if t_id != tokenizer.unk_token_id:
                        ko_result["targets"][t] = {
                            "logit": float(ko_logits[t_id].item()),
                            "prob": float(ko_probs[t_id].item()),
                            "delta_logit": float(ko_logits[t_id].item()) - float(base_logits[t_id].item())
                        }
                return ko_result

            def print_knockout_result(ko_result: Dict[str, Any]) -> None:
                print(
                    f"  - Top prediction: '{ko_result['top_token']}' "
                    f"(prob: {ko_result['top_prob']:.4f}, logit: {ko_result['top_logit']:.2f})"
                )
                for t, info in ko_result.get("targets", {}).items():
                    t_rep = t.replace("\n", "\\n").replace("\r", "\\r")
                    print(
                        f"  - Target '{t_rep}': prob: {info['prob']:.4f}, "
                        f"logit: {info['logit']:.2f} (delta: {info['delta_logit']:+.2f})"
                    )
                norms = ko_result.get("component_activation_norms", {})
                if norms:
                    print(f"  [diagnostic] Pre-knockout {ko_result['knockout_component']} output norms:")
                    for layer in ko_result.get("layers", []):
                        rows = norms.get(layer, [])
                        if not rows:
                            continue
                        if len(rows) <= 8:
                            detail = ", ".join(
                                f"{r['position']}:{r['token']!r} l2={r['l2_norm']:.2f}"
                                for r in rows
                            )
                            print(f"    L{layer}: {detail}")
                        else:
                            l2_vals = [float(r["l2_norm"]) for r in rows]
                            max_row = rows[int(np.argmax(l2_vals))]
                            print(
                                f"    L{layer}: mean_l2={float(np.mean(l2_vals)):.2f}, "
                                f"max_l2={float(max(l2_vals)):.2f} "
                                f"at {max_row['position']}:{max_row['token']!r}"
                            )

            print(
                f"\n[3/3] Running Full {args.knockout_component.upper()} Knockout "
                f"(zeroing {args.knockout_component} output at positions '{args.positions}' for layers {args.layers})..."
            )
            ko_result = run_component_knockout(args.layers, show_positions=True)
            print_knockout_result(ko_result)
            results["full_knockout"] = ko_result

            if args.layer_scan:
                print(f"\n[diagnostic] Per-layer {args.knockout_component.upper()} knockout scan:")
                layer_scan_results = []
                for layer in args.layers:
                    layer_res = run_component_knockout([layer])
                    layer_scan_results.append(layer_res)
                    print(f"\n  --- Layer {layer} only ---")
                    print_knockout_result(layer_res)
                results["layer_scan"] = layer_scan_results

            if args.position_scan:
                scan_inputs = tokenizer(args.prompt, return_tensors="pt")
                scan_positions = resolve_positions(args.positions, scan_inputs["input_ids"].shape[1])
                input_ids = scan_inputs["input_ids"][0].tolist()
                scan_records = token_position_records(tokenizer, input_ids, scan_positions)
                print(f"\n[diagnostic] Per-position {args.knockout_component.upper()} knockout scan:")
                position_scan_results = []
                for rec in scan_records:
                    pos = int(rec["position"])
                    pos_res = run_component_knockout(args.layers, position_spec_override=str(pos))
                    position_scan_results.append(pos_res)
                    print(f"\n  --- Position {pos}: {rec['token']!r} only ---")
                    print_knockout_result(pos_res)
                results["position_scan"] = position_scan_results

        # --scan: progressive ablation by attribution magnitude
        elif args.scan:
            print(f"\n[3/3] Running Progressive Ablation Scan...")
            # Build list of (layer, feature, abs_attribution) from graph nodes
            scored_features = []
            for node in graph_nodes:
                node_id = node.get("id", "")
                m = re.match(r"layer_(\d+)_feature_(\d+)", node_id)
                if m:
                    layer_idx = int(m.group(1))
                    if layer_idx not in args.layers:
                        continue
                    feat_idx = int(m.group(2))
                    signed_attribution = float(node.get("attribution", 0.0))
                    if args.graph_feature_sign == "positive" and signed_attribution <= 0:
                        continue
                    if args.graph_feature_sign == "negative" and signed_attribution >= 0:
                        continue
                    attribution = abs(signed_attribution)
                    scored_features.append((attribution, layer_idx, feat_idx))
            # Sort descending by absolute attribution
            scored_features.sort(key=lambda x: x[0], reverse=True)
            total_feats = len(scored_features)
            scan_levels = [n for n in [10, 25, 50, 100, total_feats] if n <= total_feats]
            if total_feats not in scan_levels:
                scan_levels.append(total_feats)
            # De-duplicate and ensure ascending
            scan_levels = sorted(set(scan_levels))

            print(f"  Total features from graph: {total_feats}")
            print(f"  Scan levels: {scan_levels}")
            results["scan"] = []

            for level in scan_levels:
                # Build features_dict for the top-N features
                level_features: Dict[int, List[int]] = {}
                for _, l_idx, f_idx in scored_features[:level]:
                    level_features.setdefault(l_idx, []).append(f_idx)

                level_res = run_inhibition_intervention(
                    model, tokenizer, args.prompt, args.layers, saes, level_features, target_tokens, args.edit_strength, args.positions
                )
                print(f"\n  --- Top-{level} features ablated ---")
                print(f"  Top prediction: '{level_res['top_token']}' (prob: {level_res['top_prob']:.4f}, logit: {level_res['top_logit']:.2f})")
                for t, info in level_res.get("targets", {}).items():
                    t_id = tokenizer.convert_tokens_to_ids(t)
                    if t_id == tokenizer.unk_token_id:
                        ids = tokenizer.encode(t, add_special_tokens=False)
                        if ids:
                            t_id = ids[0]
                    if t_id != tokenizer.unk_token_id:
                        delta = info["logit"] - float(base_logits[t_id].item())
                        t_rep = t.replace("\n", "\\n").replace("\r", "\\r")
                        print(f"  Target '{t_rep}': logit: {info['logit']:.2f} (delta from clean: {delta:+.2f})")

                results["scan"].append({
                    "level": level,
                    "features_ablated": level,
                    "result": level_res
                })

        else:
            # Run actual inhibition intervention (normal mode)
            print(f"\n[3/3] Running Inhibition Intervention (inhibited features: {features_dict})...")
            inter_res = run_inhibition_intervention(
                model, tokenizer, args.prompt, args.layers, saes, features_dict, target_tokens, args.edit_strength, args.positions
            )
            results["intervention"] = inter_res
            print(f"  - Top prediction: '{inter_res['top_token']}' (prob: {inter_res['top_prob']:.4f}, logit: {inter_res['top_logit']:.2f})")
            for t, info in inter_res.get("targets", {}).items():
                t_rep = t.replace("\n", "\\n").replace("\r", "\\r")
                print(f"  - Target '{t_rep}': prob: {info['prob']:.4f}, logit: {info['logit']:.2f}")

            # Diagnostic: warn about targeted features that were inactive (ablating them is a no-op).
            print("\n[diagnostic] Pre-ablation activation of targeted features (0.0 => ablation has NO effect):")
            feat_acts = inter_res.get("feature_activations", {})
            any_dead = False
            for layer in args.layers:
                for f_idx in features_dict.get(layer, []) or []:
                    act_info = feat_acts.get(layer, {}).get(int(f_idx), {})
                    if isinstance(act_info, dict):
                        max_act = float(act_info.get("max", 0.0))
                        mean_act = float(act_info.get("mean", 0.0))
                        by_pos = act_info.get("by_position", [])
                    else:
                        max_act = float(act_info or 0.0)
                        mean_act = max_act
                        by_pos = []
                    flag = "" if max_act > 0 else "  <-- INACTIVE (no-op)"
                    if max_act <= 0:
                        any_dead = True
                    print(f"  - L{layer} F{f_idx}: max={max_act:.4f}, mean={mean_act:.4f}{flag}")
                    if by_pos:
                        compact = ", ".join(
                            f"{p['position']}:{p['token']!r}={p['activation']:.3g}"
                            for p in by_pos[:24]
                        )
                        suffix = " ..." if len(by_pos) > 24 else ""
                        print(f"      by position: {compact}{suffix}")
            if any_dead:
                print("  NOTE: inactive features cannot change the output. Pick features that are"
                      " actually active for this prompt (see the attribution graph nodes).")

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
        
        if args.raw_mlp_swap:
            swap_desc = "RAW MLP output at every requested layer"
        else:
            swap_desc = "FULL latent code at every requested layer" if not features_dict else f"features {features_dict}"
        print(f"\nRunning Swap-In Intervention (swapping {swap_desc} from source to target "
              f"at positions '{args.positions}'; edit_strength={args.edit_strength})...")
        inter_res = run_swap_in_intervention(
            model, tokenizer, args.source_prompt, args.prompt, args.layers, saes,
            features_dict if features_dict else None, target_tokens, args.edit_strength,
            args.raw_mlp_swap, args.positions
        )
        results["intervention"] = inter_res
        print(f"  - Top prediction: '{inter_res['top_token']}' (prob: {inter_res['top_prob']:.4f}, logit: {inter_res['top_logit']:.2f})")
        for t, info in inter_res.get("targets", {}).items():
            t_rep = t.replace("\n", "\\n").replace("\r", "\\r")
            print(f"  - Target '{t_rep}': prob: {info['prob']:.4f}, logit: {info['logit']:.2f}")

        # --layer-scan (swap mode): patch source activations one layer at a time to
        # localize which layer's swap moves the target/contrast digits the most. This
        # is the swap-mode analogue of the full-knockout layer scan and is the clean
        # way to test whether a single layer (e.g. layer 24) carries the effect.
        if args.layer_scan:
            print(f"\n[diagnostic] Per-layer swap scan (one layer patched at a time):")
            layer_scan_results = []
            for layer in args.layers:
                # Restrict the swap to this single layer: full latent code if no
                # features were named, else that layer's named features (others no-op).
                if features_dict:
                    single = {layer: features_dict.get(layer, [])}
                else:
                    single = {L: (None if L == layer else []) for L in args.layers}
                layer_res = run_swap_in_intervention(
                    model, tokenizer, args.source_prompt, args.prompt, args.layers, saes,
                    single, target_tokens, args.edit_strength, args.raw_mlp_swap, args.positions
                )
                layer_scan_results.append({"layer": layer, "result": layer_res})
                print(f"\n  --- Layer {layer} only ---")
                print(f"  Top prediction: '{layer_res['top_token']}' "
                      f"(prob: {layer_res['top_prob']:.4f}, logit: {layer_res['top_logit']:.2f})")
                for t, info in layer_res.get("targets", {}).items():
                    t_id = tokenizer.convert_tokens_to_ids(t)
                    if t_id == tokenizer.unk_token_id:
                        ids = tokenizer.encode(t, add_special_tokens=False)
                        if ids:
                            t_id = ids[0]
                    if t_id != tokenizer.unk_token_id:
                        delta = info["logit"] - float(base_logits[t_id].item())
                        t_rep = t.replace("\n", "\\n").replace("\r", "\\r")
                        print(f"  Target '{t_rep}': logit: {info['logit']:.2f} (delta from clean: {delta:+.2f})")
            results["layer_scan"] = layer_scan_results

    # Save results
    output_path = resolve_path(args.output, repo_root)
    ensure_output_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved intervention results to {output_path}")


if __name__ == "__main__":
    main()

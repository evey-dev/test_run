"""Construct a pruned dependency graph from input features through SAE features to decisive logits."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
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
        
        # Load scaling factor from metadata if available, else default to 1.0
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
        activation_label = activation_type + (f" k={top_k}" if activation_type == "topk" else "")
        print(
            f"Loaded SAE for layer {layer} (scaling factor: {scaling_factor:.4f}, "
            f"activation: {activation_label})"
        )
        
    return saes


def generate_html_vis(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], title: str) -> str:
    """Generate standalone interactive HTML utilizing vis.js."""
    # Build vis.js compatible nodes and edges
    vis_nodes = []
    vis_edges = []
    
    # Define colors for layers (gradient from light to dark for deeper layers)
    layer_colors = {
        "input": "#d9e2ec",
        "layer_4": "#c8dae8",
        "layer_8": "#bcccdc",
        "layer_12": "#adc0d4",
        "layer_16": "#9fb3c8",
        "layer_20": "#90a6bc",
        "layer_24": "#829ab1",
        "layer_28": "#728ea5",
        "layer_32": "#627d98",
        "logits": "#486581"
    }
    
    for n in nodes:
        layer = n["layer"]
        color = layer_colors.get(layer, "#f0f4f8")
        
        # Format label and tooltip
        label = n["label"]
        title_hover = f"Layer: {layer}<br>Attribution: {n['attribution']:.5e}"
        if "top_outputs" in n and n["top_outputs"]:
            title_hover += f"<br>Top Outputs: {', '.join(n['top_outputs'])}"
            
        vis_nodes.append({
            "id": n["id"],
            "label": label,
            "title": title_hover,
            "color": color,
            "shape": "box",
            "margin": 10,
            "font": {"face": "monospace", "align": "center"}
        })
        
    for e in edges:
        weight = e["weight"]
        # Thicken based on weight magnitude
        width = min(max(abs(weight) * 200, 1), 8)
        color = "#2b6cb0" if weight > 0 else "#c53030"
        
        vis_edges.append({
            "from": e["source"],
            "to": e["target"],
            "value": abs(weight),
            "title": f"Attribution weight: {weight:.5e}",
            "width": width,
            "color": color,
            "arrows": "to"
        })
        
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f7fafc;
        }}
        #header {{
            padding: 20px;
            background-color: #2d3748;
            color: white;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        #header h1 {{
            margin: 0 0 10px 0;
            font-size: 24px;
        }}
        #header p {{
            margin: 0;
            font-size: 14px;
            color: #cbd5e0;
        }}
        #container {{
            display: flex;
            height: calc(100vh - 100px);
        }}
        #network {{
            flex-grow: 1;
            height: 100%;
        }}
        #legend {{
            width: 250px;
            background-color: white;
            border-left: 1px solid #e2e8f0;
            padding: 20px;
            box-sizing: border-box;
            font-size: 14px;
            overflow-y: auto;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 10px;
            border-radius: 4px;
            border: 1px solid #cbd5e0;
        }}
        .edge-legend {{
            margin-top: 20px;
            border-top: 1px solid #e2e8f0;
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>{title}</h1>
        <p>Interactive Mechanistic Interpretability Attribution Graph</p>
    </div>
    <div id="container">
        <div id="network"></div>
        <div id="legend">
            <h3>Layers</h3>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {layer_colors['input']};"></div>
                <span>Input Tokens</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {layer_colors['layer_4']};"></div>
                <span>Layer 4 SAE</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {layer_colors['layer_8']};"></div>
                <span>Layer 8 SAE</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {layer_colors['layer_12']};"></div>
                <span>Layer 12 SAE</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {layer_colors['layer_16']};"></div>
                <span>Layer 16 SAE</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {layer_colors['layer_20']};"></div>
                <span>Layer 20 SAE</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {layer_colors['layer_24']};"></div>
                <span>Layer 24 SAE</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {layer_colors['layer_28']};"></div>
                <span>Layer 28 SAE</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {layer_colors['layer_32']};"></div>
                <span>Layer 32 SAE</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {layer_colors['logits']};"></div>
                <span>Target Logit</span>
            </div>
            
            <div class="edge-legend">
                <h3>Edge Interactions</h3>
                <div class="legend-item">
                    <div style="width: 20px; height: 4px; background-color: #2b6cb0; margin-right: 10px;"></div>
                    <span>Positive Attribution (Excitatory)</span>
                </div>
                <div class="legend-item">
                    <div style="width: 20px; height: 4px; background-color: #c53030; margin-right: 10px;"></div>
                    <span>Negative Attribution (Inhibitory)</span>
                </div>
            </div>
        </div>
    </div>
    <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(vis_nodes)});
        var edges = new vis.DataSet({json.dumps(vis_edges)});
        var container = document.getElementById('network');
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        var options = {{
            layout: {{
                hierarchical: {{
                    direction: 'LR',
                    sortMethod: 'directed',
                    nodeSpacing: 150,
                    levelSpacing: 250
                }}
            }},
            physics: {{
                hierarchicalRepulsion: {{
                    nodeSpacing: 150
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100
            }}
        }};
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>
"""
    return html_content


def generate_mermaid_vis(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], title: str) -> str:
    """Generate a Mermaid flowchart rendering of the graph."""
    lines = [
        f"### {title}",
        "```mermaid",
        "flowchart LR"
    ]
    
    # Layer grouping/subgraphs
    layers_grouped = {}
    for n in nodes:
        layer = n["layer"]
        layers_grouped.setdefault(layer, []).append(n)
        
    for layer, l_nodes in layers_grouped.items():
        sub_name = layer.replace("_", " ").title()
        lines.append(f"    subgraph {layer} [\"{sub_name}\"]")
        for n in l_nodes:
            label_clean = n["label"].replace('"', '\\"')
            lines.append(f"        {n['id']}[\"{label_clean}\"]")
        lines.append("    end")
        
    for e in edges:
        weight = e["weight"]
        style = "-->" if weight > 0 else "-.->"
        # Optional: put edge weight as label
        lines.append(f"    {e['source']} {style}|\"{weight:.2e}\"| {e['target']}")
        
    lines.append("```")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct an attribution graph from inputs to output logits")
    parser.add_argument("--prompt", required=True, help="Input prompt to run mechanistic attribution analysis on")
    parser.add_argument("--target", default=None, help="Target next token to attribute (defaults to model's top prediction)")
    parser.add_argument("--contrast-target", default=None, help="Optional contrast token; attributes target_logit - contrast_logit instead of the raw target logit")
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 8, 12, 16, 20, 24, 28, 32], help="SAE layers to construct dependency graph through")
    parser.add_argument("--top-k-nodes", type=int, default=20, help="Number of nodes to keep per layer in the pruned graph")
    parser.add_argument("--top-k-edges", type=int, default=30, help="Number of edges to keep per layer in the pruned graph")
    parser.add_argument("--model-config", default="configs/model_config.yaml", help="Path to the model config file")
    parser.add_argument("--sae-config", default="configs/sae_config.yaml", help="Path to the SAE config file")
    parser.add_argument("--output-json", default="outputs/attribution_graph.json", help="Path to save the JSON output graph")
    parser.add_argument("--output-html", default="outputs/attribution_graph.html", help="Path to save the Vis.js HTML graph")
    parser.add_argument("--output-mermaid", default="outputs/attribution_graph.md", help="Path to save the Mermaid markdown file")
    args = parser.parse_args()

    repo_root = get_repo_root()
    sae_cfg = load_yaml_config(repo_root / args.sae_config)
    sae_dir = resolve_path(sae_cfg.get("output_dir", "mechanistic_data/sae_checkpoints"), repo_root)
    hidden_size = int(sae_cfg.get("hidden_size", 2560))
    latent_dim = int(sae_cfg.get("latent_dim", 8192))
    
    print("Loading model and tokenizer...")
    model, tokenizer, model_cfg = load_model_and_tokenizer(repo_root / args.model_config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Disable gradients for all model and SAE parameters to speed up backpropagation and save VRAM
    print("Disabling gradients for all model parameters for speed and memory efficiency.")
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Loading SAE models from {sae_dir} to device: {device}...")
    saes = load_sae_models(args.layers, sae_dir, hidden_size, latent_dim, device, model.dtype)
    for layer in args.layers:
        sae_model, _ = saes[layer]
        for param in sae_model.parameters():
            param.requires_grad = False

    # 1. Prepare Inputs
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    
    # 2. Setup Hooks for Embeddings and SAE Reconstructions
    captured_embeddings = []
    captured_z = {}
    hooks = []
    
    # Hook for input embeddings
    def embed_hook(module, input_t, output_t):
        output_t.requires_grad_(True)
        output_t.retain_grad()
        captured_embeddings.append(output_t)
        return output_t
    
    # Registered to the embed_tokens block
    h_embed = model.model.embed_tokens.register_forward_hook(embed_hook)
    hooks.append(h_embed)
    
    # Hooks for SAE reconstructions at layer MLPs
    def make_reconstruction_hook(layer_idx, sae_model, scaling_factor):
        def hook_fn(module, input_t, output_t):
            # output_t is [batch_size, seq_len, hidden_size]
            # Capture and reconstruct only on the last token position
            last_token_act = output_t[:, -1, :] # [batch_size, hidden_size]
            last_token_norm = last_token_act / scaling_factor
            
            # SAE Encode
            z = sae_model.encode(last_token_norm)
            z.retain_grad()
            captured_z[layer_idx] = z
            
            # SAE Decode
            x_hat_norm = sae_model.decoder(z) + sae_model.decoder_bias
            x_hat = x_hat_norm * scaling_factor

            # Error-preserving straight-through edit at the last token position.
            # The forward VALUE stays the model's true activation (so predictions are
            # bit-for-bit identical to the clean run), while gradients still flow
            # through the SAE latents z via (x_hat - x_hat.detach()) == 0 in value.
            # This avoids stacking every hooked layer's reconstruction error into the
            # forward pass (which previously swamped the signal and collapsed the
            # prediction to a bare token). Mirrors the splice edit used in intervention.py.
            new_output = output_t.clone()
            new_output[:, -1, :] = last_token_act + (x_hat - x_hat.detach())
            return new_output
        return hook_fn

    for layer in args.layers:
        sae_model, scaling_factor = saes[layer]
        mlp_module = model.model.layers[layer].mlp
        h = mlp_module.register_forward_hook(make_reconstruction_hook(layer, sae_model, scaling_factor))
        hooks.append(h)

    # 3. Run Forward Pass to get logits and capture intermediate states
    print("Running forward pass...")
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :] # shape [vocab_size]
    
    # Clean hooks immediately after forward pass
    for h in hooks:
        h.remove()

    # Print top 5 predicted tokens
    probs = torch.softmax(logits, dim=-1)
    top_p, top_i = torch.topk(probs, 5)
    print("\nTop 5 model predictions:")
    for p, i in zip(top_p, top_i):
        print(f"  '{tokenizer.decode([i.item()])}' (prob: {p.item():.4f}, logit: {logits[i.item()].item():.2f})")
    print()

    def resolve_token_id(token_text: str) -> Tuple[int, str]:
        """Resolve a display token to the single token ID with highest current probability."""
        # Try both the raw target, target with space, and stripped target.
        targets_to_try = [token_text, " " + token_text, token_text.strip()]
        best_token_id = None
        best_prob = -1.0

        for t in targets_to_try:
            tid = tokenizer.convert_tokens_to_ids(t)
            if tid == tokenizer.unk_token_id:
                target_ids = tokenizer.encode(t, add_special_tokens=False)
                if target_ids:
                    tid = target_ids[0]
            if tid != tokenizer.unk_token_id:
                prob = probs[tid].item()
                if prob > best_prob:
                    best_prob = prob
                    best_token_id = tid

        if best_token_id is not None:
            return best_token_id, tokenizer.decode([best_token_id])
        raise ValueError(f"Could not tokenize target token: '{token_text}'")

    # Determine Target Token
    if args.target:
        target_token_id, target_token = resolve_token_id(args.target)
    else:
        # Default to top predicted token
        target_token_id = torch.argmax(logits).item()
        target_token = tokenizer.decode([target_token_id])

    contrast_token_id = None
    contrast_token = None
    if args.contrast_target:
        contrast_token_id, contrast_token = resolve_token_id(args.contrast_target)

    print(f"Target token for attribution: '{target_token}' (ID: {target_token_id}, probability: {probs[target_token_id].item():.4f})")
    if contrast_token_id is not None:
        print(f"Contrast token: '{contrast_token}' (ID: {contrast_token_id}, probability: {probs[contrast_token_id].item():.4f})")
    
    # 4. Run Backward Pass from Target Logit
    target_logit = logits[target_token_id]
    attribution_objective = target_logit
    objective_label = f"Logit: '{target_token}'"
    if contrast_token_id is not None:
        contrast_logit = logits[contrast_token_id]
        attribution_objective = target_logit - contrast_logit
        objective_label = f"Logit diff: '{target_token}' - '{contrast_token}'"
        print(
            f"Running backward pass on logit difference: "
            f"{target_logit.item():.4f} - {contrast_logit.item():.4f} = {attribution_objective.item():.4f}"
        )
    else:
        print(f"Running backward pass on logit: {target_logit.item():.4f}")
    attribution_objective.backward(retain_graph=True)
    
    # Gather node attributions and gradients
    z_dict = {}
    g_dict = {}
    
    for layer in args.layers:
        z_dict[layer] = captured_z[layer].detach().cpu()
        g_dict[layer] = captured_z[layer].grad.detach().cpu()
        
    # Input embeddings gradients
    embedding_grad = captured_embeddings[0].grad.detach().cpu() # shape [1, seq_len, hidden_size]
    embedding_val = captured_embeddings[0].detach().cpu() # shape [1, seq_len, hidden_size]
    
    # 5. Compute Node Attribution Scores
    node_attributions = {}
    node_labels = {}
    node_top_outputs = {}
    
    # Input Nodes (Tokens)
    token_list = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
    for p in range(seq_len):
        node_id = f"input_{p}"
        # Direct attribution: dot product of embedding and its gradient
        direct_attr = float((embedding_val[0, p, :] * embedding_grad[0, p, :]).sum().item())
        node_attributions[node_id] = direct_attr
        node_labels[node_id] = f"[{p}] '{token_list[p]}'"
        node_top_outputs[node_id] = []
        
    # SAE Feature Nodes
    print("Computing SAE feature attributions...")
    for layer in args.layers:
        z = z_dict[layer][0]
        g = g_dict[layer][0]
        
        # Linear contribution: z_i * g_i
        attr = z * g
        
        for i in range(latent_dim):
            node_id = f"layer_{layer}_feature_{i}"
            node_attributions[node_id] = float(attr[i].item())
            
            # Set label
            label = f"L{layer} F{i}"
            node_labels[node_id] = label

    # 6. Compute Edge Attribution Scores
    print("Tracing paths and computing edge attributions (EAP)...")
    edges = []
    
    # Layer-to-Layer Edges
    for l_idx in range(len(args.layers) - 1):
        l1 = args.layers[l_idx]
        l2 = args.layers[l_idx + 1]
        
        z_l1 = captured_z[l1]
        z_l2 = captured_z[l2]
        g_l2 = captured_z[l2].grad
        
        # Only compute gradients for active features in layer 2
        active_l2 = [j for j in range(latent_dim) if z_dict[l2][0, j].item() > 0 and abs(g_dict[l2][0, j].item()) > 1e-5]
        active_l1 = [i for i in range(latent_dim) if z_dict[l1][0, i].item() > 0]
        
        print(f"Tracing edges from Layer {l1} ({len(active_l1)} active) to Layer {l2} ({len(active_l2)} active)...")
        
        for j in active_l2:
            # Gradient of z_l2[j] with respect to z_l1
            grad_z_l1 = torch.autograd.grad(
                outputs=z_l2[0, j],
                inputs=z_l1,
                retain_graph=True,
                only_inputs=True
            )[0][0] # shape [latent_dim]
            
            grad_z_l1_cpu = grad_z_l1.detach().cpu()
            del grad_z_l1
            
            for i in active_l1:
                val = z_dict[l1][0, i] * grad_z_l1_cpu[i] * g_dict[l2][0, j]
                val_item = float(val.item())
                if abs(val_item) > 1e-6:
                    edges.append({
                        "source": f"layer_{l1}_feature_{i}",
                        "target": f"layer_{l2}_feature_{j}",
                        "weight": val_item
                    })
                    
    # Input to First SAE Layer Edges
    first_layer = args.layers[0]
    z_first = captured_z[first_layer]
    active_first = [j for j in range(latent_dim) if z_dict[first_layer][0, j].item() > 0 and abs(g_dict[first_layer][0, j].item()) > 1e-5]
    print(f"Tracing edges from Input ({seq_len} tokens) to Layer {first_layer} ({len(active_first)} active)...")

    for j in active_first:
        grad_embed = torch.autograd.grad(
            outputs=z_first[0, j],
            inputs=captured_embeddings[0],
            retain_graph=True,
            only_inputs=True
        )[0][0] # shape [seq_len, hidden_size]

        grad_embed_cpu = grad_embed.detach().cpu()
        del grad_embed

        for p in range(seq_len):
            # dot product of embedding and gradient wrt embedding, scaled by downstream feature gradient
            dot = (embedding_val[0, p, :] * grad_embed_cpu[p, :]).sum().item()
            val_item = float(dot * g_dict[first_layer][0, j].item())
            if abs(val_item) > 1e-6:
                edges.append({
                    "source": f"input_{p}",
                    "target": f"layer_{first_layer}_feature_{j}",
                    "weight": val_item
                })

    # Add terminal target logit node
    node_attributions["target_logit"] = float(attribution_objective.item())
    node_labels["target_logit"] = objective_label
    node_top_outputs["target_logit"] = []
    
    # Last SAE Layer to Logit Edges (direct attribution is the node attribution score)
    last_layer = args.layers[-1]
    active_last = [i for i in range(latent_dim) if z_dict[last_layer][0, i].item() > 0]
    for i in active_last:
        val_item = float(z_dict[last_layer][0, i].item() * g_dict[last_layer][0, i].item())
        if abs(val_item) > 1e-6:
            edges.append({
                "source": f"layer_{last_layer}_feature_{i}",
                "target": "target_logit",
                "weight": val_item
            })

    # 7. Graph Pruning
    print("Pruning graph to select top nodes and edges...")
    # Keep top-K nodes per layer based on absolute attribution score
    pruned_nodes_set = set(["target_logit"])
    
    # Input nodes
    input_nodes = [f"input_{p}" for p in range(seq_len)]
    input_nodes.sort(key=lambda n: abs(node_attributions[n]), reverse=True)
    pruned_nodes_set.update(input_nodes[:args.top_k_nodes])
    
    # SAE Layers nodes
    for layer in args.layers:
        layer_nodes = [f"layer_{layer}_feature_{i}" for i in range(latent_dim)]
        layer_nodes.sort(key=lambda n: abs(node_attributions[n]), reverse=True)
        # Only include nodes that are actually active (attribution != 0)
        active_layer_nodes = [n for n in layer_nodes if node_attributions[n] != 0]
        pruned_nodes_set.update(active_layer_nodes[:args.top_k_nodes])
        
    # Prune edges to keep only those connecting the pruned nodes
    pruned_edges = []
    for e in edges:
        if e["source"] in pruned_nodes_set and e["target"] in pruned_nodes_set:
            pruned_edges.append(e)
            
    # Also prune nodes that are completely isolated (degree = 0) except target logit
    connected_nodes = set(["target_logit"])
    for e in pruned_edges:
        connected_nodes.add(e["source"])
        connected_nodes.add(e["target"])
        
    # Now compute DLA interpretations only for the selected pruned feature nodes!
    print("Computing Direct Logit Attribution (DLA) for selected pruned nodes...")
    node_top_outputs = {}
    for n_id in pruned_nodes_set:
        node_top_outputs[n_id] = []
        if n_id in connected_nodes and n_id.startswith("layer_"):
            parts = n_id.split("_")
            layer = int(parts[1])
            f_idx = int(parts[3])
            
            sae_model, _ = saes[layer]
            z_val = z_dict[layer][0, f_idx].item()
            
            if z_val > 0:
                decoder_vec = sae_model.decoder.weight[:, f_idx].detach() # [hidden_size]
                dla_logits = torch.matmul(model.lm_head.weight, decoder_vec.to(device=model.device, dtype=model.dtype))
                top_vals, top_idx = torch.topk(dla_logits, k=3)
                top_tokens = []
                for idx in top_idx:
                    tok = tokenizer.decode([idx.item()]).strip()
                    top_tokens.append(f"'{tok}'")
                node_top_outputs[n_id] = top_tokens

    pruned_nodes_list = []
    for n_id in pruned_nodes_set:
        if n_id in connected_nodes:
            layer = "logits" if n_id == "target_logit" else ("input" if n_id.startswith("input_") else f"layer_{n_id.split('_')[1]}")
            pruned_nodes_list.append({
                "id": n_id,
                "label": node_labels[n_id],
                "layer": layer,
                "attribution": node_attributions[n_id],
                "top_outputs": node_top_outputs.get(n_id, [])
            })
            
    # Sort nodes by layer and attribution magnitude for structured viewing
    layer_order = {"input": 0, "layer_4": 1, "layer_8": 2, "layer_12": 3, "layer_16": 4, "layer_20": 5, "layer_24": 6, "layer_28": 7, "layer_32": 8, "logits": 9}
    pruned_nodes_list.sort(key=lambda x: (layer_order.get(x["layer"], 99), -abs(x["attribution"])))

    # Save outputs
    output_dir = ensure_output_dir(resolve_path(args.output_json, repo_root).parent)
    
    # 1. JSON Export
    graph_payload = {
        "prompt": args.prompt,
        "target": target_token,
        "target_prob": float(probs[target_token_id].item()),
        "contrast_target": contrast_token,
        "contrast_target_prob": float(probs[contrast_token_id].item()) if contrast_token_id is not None else None,
        "attribution_objective": objective_label,
        "attribution_objective_value": float(attribution_objective.item()),
        "nodes": pruned_nodes_list,
        "edges": pruned_edges
    }
    
    json_path = resolve_path(args.output_json, repo_root)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(graph_payload, fh, indent=2)
    print(f"Saved pruned graph JSON to {json_path}")
    
    # 2. Mermaid Diagram
    graph_title = f"Attribution Graph: {args.prompt} -> {objective_label}"
    mermaid_content = generate_mermaid_vis(pruned_nodes_list, pruned_edges, graph_title)
    mermaid_path = resolve_path(args.output_mermaid, repo_root)
    with open(mermaid_path, "w", encoding="utf-8") as fh:
        fh.write(mermaid_content)
    print(f"Saved Mermaid chart to {mermaid_path}")
    
    # 3. HTML vis.js Visualization
    html_content = generate_html_vis(pruned_nodes_list, pruned_edges, graph_title)
    html_path = resolve_path(args.output_html, repo_root)
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html_content)
    print(f"Saved interactive HTML visualization to {html_path}")

    # 4. Emit a ready-to-paste --features string for intervention.py, built from the
    # graph's own top feature nodes. This closes the loop between the graph and the
    # intervention step so you inhibit features that are actually active for this prompt
    # (rather than guessing indices).
    features_for_intervention: Dict[int, List[int]] = {}
    for n in pruned_nodes_list:
        if n["layer"].startswith("layer_"):
            layer_num = int(n["layer"].split("_")[1])
            feat_idx = int(n["id"].split("_")[3])
            features_for_intervention.setdefault(layer_num, []).append(feat_idx)
    features_json = json.dumps({str(k): v for k, v in sorted(features_for_intervention.items())})
    print("\nSuggested intervention on the graph's top features (features that are active for this prompt):")
    print(f"  --features '{features_json}'")


if __name__ == "__main__":
    main()

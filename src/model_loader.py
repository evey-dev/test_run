"""Load the local Qwen3 model and tokenizer for inference.

This module exposes helpers to set seeds and load a local Hugging Face causal
language model from a YAML configuration file. It is used by the baseline
runner and other evaluation scripts.
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


SEED = 787


def set_seed(seed: int = SEED) -> None:
    """Set all relevant seeds for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_config_path: str | None = None) -> Tuple[torch.nn.Module, Any, Dict[str, Any]]:
    """Load a local Hugging Face causal LM and return the model, tokenizer, and config."""
    if model_config_path is None:
        model_config_path = "configs/model_config.yaml"

    with open(model_config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    model_id = cfg.get("model_id", "./models/Qwen3-4B-Instruct")
    revision = cfg.get("revision", "main")
    torch_dtype = getattr(torch, cfg.get("torch_dtype", "bfloat16"))
    device_map = cfg.get("device_map", "auto")
    max_new_tokens = cfg.get("max_new_tokens", 50)
    do_sample = bool(cfg.get("do_sample", False))
    temperature = float(cfg.get("temperature", 1.0))

    set_seed(cfg.get("seed", SEED))

    if not os.path.exists(model_id):
        if model_id in ("./models/Qwen3-4B-Instruct", "models/Qwen3-4B-Instruct"):
            print(f"Local model path '{model_id}' not found. Falling back to Hugging Face Hub: Qwen/Qwen3-4B-Instruct-2507")
            model_id = "Qwen/Qwen3-4B-Instruct-2507"
        else:
            is_hf_repo = len(model_id.split("/")) == 2 and not model_id.startswith(".")
            if not is_hf_repo:
                raise FileNotFoundError(f"Model directory or Hugging Face repository not found: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    cfg = {
        **cfg,
        "model_dir": model_id,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
    }
    return model, tokenizer, cfg

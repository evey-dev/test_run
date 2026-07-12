"""Localise carry information after balancing away the answer digit.

This experiment differs from the single-graph carry screen in two ways:

1. Candidate SAE features are ranked by carry-versus-no-carry activation
   differences within strata that share the same predicted tens digit.
2. The frozen panel is evaluated causally on disjoint matched arithmetic pairs.

The script also fits a simple output-digit-conditioned linear direction to the
raw final-token MLP output at each SAE layer. This diagnostic separates a weak
activation site from a sparse-basis mismatch. Confirmation observations never
influence feature ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from src.data_utils import get_repo_root, resolve_path
from src.heldout_validation import (
    LAYERS,
    bootstrap_mean_ci,
    condition_from_result,
    first_token_id,
    generated_math_cases,
    load_domain_saes,
    suppress_output,
)
from src.intervention import run_inhibition_intervention
from src.model_loader import load_model_and_tokenizer


Feature = Tuple[int, int]


def checkpoint_payload(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(output_path)


def feature_key(feature: Feature) -> str:
    return f"L{int(feature[0])}F{int(feature[1])}"


def feature_dict(features: Iterable[Feature]) -> Dict[int, List[int]]:
    selected: Dict[int, List[int]] = {}
    for layer, feature in features:
        selected.setdefault(int(layer), []).append(int(feature))
    return {layer: sorted(set(values)) for layer, values in sorted(selected.items())}


def arithmetic_case_key(case: Dict[str, Any]) -> str:
    return f"{int(case['source_a'])}+{int(case['b'])}->{int(case['target_a'])}+{int(case['b'])}"


def collect_case_keys(value: Any) -> set[str]:
    """Recursively recover arithmetic case keys from prior result JSON."""
    keys: set[str] = set()
    if isinstance(value, dict):
        if isinstance(value.get("case_key"), str):
            keys.add(value["case_key"])
        required = {"source_a", "target_a", "b"}
        if required.issubset(value):
            try:
                keys.add(arithmetic_case_key(value))
            except (TypeError, ValueError):
                pass
        for child in value.values():
            keys.update(collect_case_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(collect_case_keys(child))
    return keys


def load_excluded_case_keys(paths: Sequence[Path]) -> Tuple[set[str], List[Dict[str, Any]]]:
    excluded: set[str] = set()
    sources = []
    for path in paths:
        if not path.exists():
            sources.append({"path": str(path), "exists": False, "case_count": 0})
            continue
        with path.open("r", encoding="utf-8") as handle:
            keys = collect_case_keys(json.load(handle))
        excluded.update(keys)
        sources.append({"path": str(path), "exists": True, "case_count": len(keys)})
    return excluded, sources


def hash_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def batched(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def observation_templates(cases: Sequence[Dict[str, Any]], tokenizer) -> List[Dict[str, Any]]:
    records = []
    for case in cases:
        key = arithmetic_case_key(case)
        carry_id = first_token_id(tokenizer, case["correct_digit"])
        no_carry_id = first_token_id(tokenizer, case["dropped_carry_digit"])
        records.extend(
            [
                {
                    "case_key": key,
                    "condition": "carry",
                    "carry_label": 1,
                    "output_digit": str(case["correct_digit"]),
                    "prompt": case["target_prompt"],
                    "correct_id": carry_id,
                    "contrast_id": no_carry_id,
                },
                {
                    "case_key": key,
                    "condition": "no_carry",
                    "carry_label": 0,
                    "output_digit": str(case["dropped_carry_digit"]),
                    "prompt": case["source_prompt"],
                    "correct_id": no_carry_id,
                    "contrast_id": carry_id,
                },
            ]
        )
    return records


def screen_baselines(
    model,
    tokenizer,
    cases: Sequence[Dict[str, Any]],
    batch_size: int,
) -> List[Dict[str, Any]]:
    templates = observation_templates(cases, tokenizer)
    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for batch_index, batch in enumerate(batched(templates, batch_size), start=1):
        prompts = [row["prompt"] for row in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
            probabilities = torch.softmax(logits.float(), dim=-1)
            top_probabilities, top_ids = probabilities.max(dim=-1)

        for index, template in enumerate(batch):
            correct_id = int(template["correct_id"])
            contrast_id = int(template["contrast_id"])
            top_id = int(top_ids[index].item())
            record = {
                "correct_id": correct_id,
                "contrast_id": contrast_id,
                "correct_token": tokenizer.decode([correct_id]),
                "contrast_token": tokenizer.decode([contrast_id]),
                "correct_logit": float(logits[index, correct_id].item()),
                "contrast_logit": float(logits[index, contrast_id].item()),
                "gap": float((logits[index, correct_id] - logits[index, contrast_id]).item()),
                "top_id": top_id,
                "top_token": tokenizer.decode([top_id]),
                "top_probability": float(top_probabilities[index].item()),
                "top_is_correct": top_id == correct_id,
                "token_count": int(inputs["attention_mask"][index].sum().item()),
            }
            results[(template["case_key"], template["condition"])] = record
        print(f"[baseline batch {batch_index}] screened {len(batch)} prompts")

    prepared = []
    for case in cases:
        row = dict(case)
        key = arithmetic_case_key(case)
        carry = results[(key, "carry")]
        no_carry = results[(key, "no_carry")]
        row["case_key"] = key
        row["carry_baseline"] = carry
        row["no_carry_baseline"] = no_carry
        row["eligible"] = bool(
            carry["top_is_correct"]
            and no_carry["top_is_correct"]
            and carry["token_count"] == no_carry["token_count"]
        )
        if not row["eligible"]:
            row["ineligible_reason"] = (
                "carry/no-carry baseline incorrect or token lengths differ"
            )
        prepared.append(row)
    return prepared


def pair_balance(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Counter[Tuple[str, int]] = Counter()
    for case in cases:
        counts[(str(case["correct_digit"]), 1)] += 1
        counts[(str(case["dropped_carry_digit"]), 0)] += 1
    common = sorted(
        digit
        for digit in {digit for digit, _ in counts}
        if counts[(digit, 1)] > 0 and counts[(digit, 0)] > 0
    )
    matched = sum(min(counts[(digit, 1)], counts[(digit, 0)]) for digit in common)
    return {
        "common_output_digits": common,
        "common_digit_count": len(common),
        "matched_observations_per_class": int(matched),
        "counts": {
            digit: {"carry": counts[(digit, 1)], "no_carry": counts[(digit, 0)]}
            for digit in sorted({digit for digit, _ in counts})
        },
    }


def choose_balanced_split(
    eligible_cases: Sequence[Dict[str, Any]],
    discovery_pairs: int,
    confirmation_pairs: int,
    seed: int,
    trials: int = 4000,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    required = discovery_pairs + confirmation_pairs
    if len(eligible_cases) < required:
        raise ValueError(
            f"Only {len(eligible_cases)} eligible pairs are available; {required} are required"
        )

    rng = np.random.default_rng(seed)
    best = None
    best_score = None
    for _ in range(trials):
        order = rng.permutation(len(eligible_cases))[:required]
        discovery = [eligible_cases[int(index)] for index in order[:discovery_pairs]]
        confirmation = [eligible_cases[int(index)] for index in order[discovery_pairs:]]
        discovery_balance = pair_balance(discovery)
        confirmation_balance = pair_balance(confirmation)
        score = (
            min(
                discovery_balance["common_digit_count"],
                confirmation_balance["common_digit_count"],
            ),
            discovery_balance["common_digit_count"]
            + confirmation_balance["common_digit_count"],
            min(
                discovery_balance["matched_observations_per_class"],
                confirmation_balance["matched_observations_per_class"],
            ),
            discovery_balance["matched_observations_per_class"]
            + confirmation_balance["matched_observations_per_class"],
        )
        if best_score is None or score > best_score:
            best_score = score
            best = (discovery, confirmation, discovery_balance, confirmation_balance)

    assert best is not None
    discovery, confirmation, discovery_balance, confirmation_balance = best
    if min(discovery_balance["common_digit_count"], confirmation_balance["common_digit_count"]) < 5:
        raise ValueError("Could not construct splits with at least five common output-digit strata")
    return discovery, confirmation, {
        "optimisation_trials": trials,
        "selection_used_only_baseline_eligibility_and_output_digits": True,
        "discovery": discovery_balance,
        "confirmation": confirmation_balance,
    }


def public_case(case: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "case_key",
        "target_a",
        "source_a",
        "b",
        "target_sum",
        "source_sum",
        "correct_digit",
        "dropped_carry_digit",
        "target_prompt",
        "source_prompt",
        "absent_from_sae_corpus",
        "eligible",
        "ineligible_reason",
        "carry_baseline",
        "no_carry_baseline",
    ]
    return {key: case[key] for key in keep if key in case}


def collect_activation_dataset(
    model,
    tokenizer,
    saes,
    cases: Sequence[Dict[str, Any]],
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[int, np.ndarray]]:
    records = observation_templates(cases, tokenizer)
    latent_chunks: Dict[int, List[np.ndarray]] = {layer: [] for layer in LAYERS}
    raw_chunks: Dict[int, List[np.ndarray]] = {layer: [] for layer in LAYERS}
    captured: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    hooks = []

    def make_hook(layer: int):
        sae, scaling_factor = saes[layer]

        def hook_fn(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            raw = tensor[:, -1, :]
            with torch.no_grad():
                latent = sae.encode(raw / scaling_factor)
            captured[layer] = (
                raw.detach().float().cpu().numpy().astype(np.float16),
                latent.detach().float().cpu().numpy().astype(np.float16),
            )

        return hook_fn

    for layer in LAYERS:
        hooks.append(model.model.layers[layer].mlp.register_forward_hook(make_hook(layer)))

    tokenizer.padding_side = "left"
    try:
        for batch_index, batch in enumerate(batched(records, batch_size), start=1):
            captured.clear()
            prompts = [row["prompt"] for row in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                model(**inputs)
            missing = set(LAYERS).difference(captured)
            if missing:
                raise RuntimeError(f"MLP hooks did not fire for layers {sorted(missing)}")
            for layer in LAYERS:
                raw, latent = captured[layer]
                raw_chunks[layer].append(raw)
                latent_chunks[layer].append(latent)
            print(f"[activation batch {batch_index}] captured {len(batch)} prompts")
    finally:
        for hook in hooks:
            hook.remove()

    raw_by_layer = {
        layer: np.concatenate(raw_chunks[layer], axis=0) for layer in LAYERS
    }
    latent_by_layer = [np.concatenate(latent_chunks[layer], axis=0) for layer in LAYERS]
    latent_matrix = np.concatenate(latent_by_layer, axis=1)
    if latent_matrix.shape[0] != len(records):
        raise RuntimeError("Activation rows do not align with observation records")
    return records, latent_matrix, raw_by_layer


def save_activation_cache(
    path: Path,
    signature: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    selected_cases: Sequence[Dict[str, Any]],
    latent_matrix: np.ndarray,
    raw_by_layer: Dict[int, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    arrays: Dict[str, Any] = {
        "signature": np.asarray(json.dumps(signature, sort_keys=True)),
        "records": np.asarray(json.dumps(list(records))),
        "selected_cases": np.asarray(json.dumps([public_case(case) for case in selected_cases])),
        "latent": latent_matrix.astype(np.float16),
    }
    arrays.update({f"raw_layer_{layer}": values.astype(np.float16) for layer, values in raw_by_layer.items()})
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_activation_cache(
    path: Path,
    signature: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], np.ndarray, Dict[int, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        stored_signature = json.loads(str(payload["signature"].item()))
        if stored_signature != signature:
            raise ValueError(
                f"Activation cache {path} uses a different protocol; delete it or choose another path"
            )
        records = json.loads(str(payload["records"].item()))
        selected_cases = json.loads(str(payload["selected_cases"].item()))
        latent = payload["latent"].astype(np.float32)
        raw = {layer: payload[f"raw_layer_{layer}"].astype(np.float32) for layer in LAYERS}
    return records, selected_cases, latent, raw


def indices_for_pairs(records: Sequence[Dict[str, Any]], keys: Iterable[str]) -> np.ndarray:
    selected = set(keys)
    return np.asarray(
        [index for index, row in enumerate(records) if row["case_key"] in selected],
        dtype=int,
    )


def common_digit_groups(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[int, np.ndarray]]:
    grouped: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: {0: [], 1: []})
    for index, row in enumerate(records):
        grouped[str(row["output_digit"])][int(row["carry_label"])].append(index)
    return {
        digit: {
            label: np.asarray(indices, dtype=int)
            for label, indices in by_label.items()
        }
        for digit, by_label in sorted(grouped.items())
        if by_label[0] and by_label[1]
    }


def conditional_auc(scores: np.ndarray, records: Sequence[Dict[str, Any]]) -> float:
    groups = common_digit_groups(records)
    aucs = []
    for by_label in groups.values():
        carry = scores[by_label[1]][:, None]
        no_carry = scores[by_label[0]][None, :]
        aucs.append(float(np.mean(carry > no_carry) + 0.5 * np.mean(carry == no_carry)))
    return float(np.mean(aucs)) if aucs else float("nan")


def conditional_score_summary(
    scores: np.ndarray,
    records: Sequence[Dict[str, Any]],
    seed: int,
    bootstrap_samples: int = 2000,
) -> Dict[str, Any]:
    groups = common_digit_groups(records)
    if not groups:
        return {"observation_count": len(records), "common_output_digits": []}

    effects = []
    for by_label in groups.values():
        effects.append(float(scores[by_label[1]].mean() - scores[by_label[0]].mean()))
    observed = float(np.mean(effects))

    rng = np.random.default_rng(seed)
    boot = np.empty(bootstrap_samples, dtype=float)
    group_values = list(groups.values())
    for sample_index in range(bootstrap_samples):
        stratum_effects = []
        for by_label in group_values:
            carry = scores[by_label[1]]
            no_carry = scores[by_label[0]]
            carry_sample = rng.choice(carry, size=len(carry), replace=True)
            no_carry_sample = rng.choice(no_carry, size=len(no_carry), replace=True)
            stratum_effects.append(float(carry_sample.mean() - no_carry_sample.mean()))
        boot[sample_index] = float(np.mean(stratum_effects))

    return {
        "observation_count": len(records),
        "common_output_digits": list(groups),
        "common_output_digit_count": len(groups),
        "mean_within_digit_carry_minus_no_carry": observed,
        "bootstrap_95_ci_mean_within_digit_difference": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "output_digit_conditioned_auc": conditional_auc(scores, records),
        "stratum_effects": {
            digit: float(scores[by_label[1]].mean() - scores[by_label[0]].mean())
            for digit, by_label in groups.items()
        },
    }


def fit_conditional_direction(
    matrix: np.ndarray,
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    matrix = matrix.astype(np.float32)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-5] = 1.0
    standardised = (matrix - mean) / scale
    groups = common_digit_groups(records)
    digit_centres = {
        digit: standardised[np.concatenate([by_label[0], by_label[1]])].mean(axis=0)
        for digit, by_label in groups.items()
    }
    residual = standardised.copy()
    for index, row in enumerate(records):
        centre = digit_centres.get(str(row["output_digit"]))
        if centre is not None:
            residual[index] -= centre
    labels = np.asarray([int(row["carry_label"]) for row in records], dtype=int)
    direction = residual[labels == 1].mean(axis=0) - residual[labels == 0].mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("Conditional carry direction has zero norm")
    direction /= norm
    scores = residual @ direction
    threshold = float((scores[labels == 1].mean() + scores[labels == 0].mean()) / 2)
    return {
        "mean": mean,
        "scale": scale,
        "digit_centres": digit_centres,
        "direction": direction,
        "threshold": threshold,
        "training_scores": scores,
    }


def apply_conditional_direction(
    fit: Dict[str, Any],
    matrix: np.ndarray,
    records: Sequence[Dict[str, Any]],
) -> np.ndarray:
    standardised = (matrix.astype(np.float32) - fit["mean"]) / fit["scale"]
    for index, row in enumerate(records):
        centre = fit["digit_centres"].get(str(row["output_digit"]))
        if centre is not None:
            standardised[index] -= centre
    return standardised @ fit["direction"]


def rank_carry_features(
    discovery_matrix: np.ndarray,
    discovery_records: Sequence[Dict[str, Any]],
    layers: Sequence[int],
    minimum_active_fraction: float,
    minimum_positive_stratum_fraction: float,
) -> Tuple[List[int], Dict[str, np.ndarray], List[Dict[str, Any]]]:
    matrix = discovery_matrix.astype(np.float32)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    safe_scale = scale.copy()
    safe_scale[safe_scale < 1e-5] = 1.0
    standardised = (matrix - mean) / safe_scale
    groups = common_digit_groups(discovery_records)
    differences = np.stack(
        [
            standardised[by_label[1]].mean(axis=0)
            - standardised[by_label[0]].mean(axis=0)
            for by_label in groups.values()
        ],
        axis=0,
    )
    effect = differences.mean(axis=0)
    positive_strata = (differences > 0).sum(axis=0)
    required_positive = max(1, math.ceil(minimum_positive_stratum_fraction * len(groups)))
    labels = np.asarray([int(row["carry_label"]) for row in discovery_records], dtype=int)
    carry_active = (matrix[labels == 1] > 1e-6).mean(axis=0)
    no_carry_active = (matrix[labels == 0] > 1e-6).mean(axis=0)
    candidate_mask = (
        (effect > 0)
        & (carry_active >= minimum_active_fraction)
        & (positive_strata >= required_positive)
        & (scale >= 1e-5)
    )
    candidates = np.flatnonzero(candidate_mask)
    if len(candidates) < 20:
        raise ValueError(
            f"Only {len(candidates)} balanced carry-associated candidates passed the fixed filters"
        )
    ordering = candidates[
        np.lexsort(
            (
                candidates,
                -carry_active[candidates],
                -positive_strata[candidates],
                -effect[candidates],
            )
        )
    ]

    latent_dim = matrix.shape[1] // len(layers)
    if latent_dim * len(layers) != matrix.shape[1]:
        raise ValueError("Latent matrix width is not divisible by the number of layers")

    def record_for_column(column: int, rank: int) -> Dict[str, Any]:
        layer_index, feature = divmod(int(column), latent_dim)
        layer = int(layers[layer_index])
        return {
            "rank": rank,
            "key": feature_key((layer, feature)),
            "layer": layer,
            "feature": feature,
            "column": int(column),
            "conditional_standardised_effect": float(effect[column]),
            "positive_output_digit_strata": int(positive_strata[column]),
            "total_output_digit_strata": len(groups),
            "carry_active_fraction": float(carry_active[column]),
            "no_carry_active_fraction": float(no_carry_active[column]),
        }

    records = [record_for_column(int(column), rank) for rank, column in enumerate(ordering[:200], 1)]
    statistics = {
        "mean": mean,
        "scale": safe_scale,
        "effect": effect,
        "positive_strata": positive_strata,
        "carry_active": carry_active,
        "no_carry_active": no_carry_active,
        "latent_dim": np.asarray(latent_dim),
        "required_positive_strata": np.asarray(required_positive),
    }
    return [int(column) for column in ordering], statistics, records


def column_to_feature(column: int, layers: Sequence[int], latent_dim: int) -> Feature:
    layer_index, feature = divmod(int(column), int(latent_dim))
    return int(layers[layer_index]), int(feature)


def panel_scores(
    matrix: np.ndarray,
    columns: Sequence[int],
    discovery_statistics: Dict[str, np.ndarray],
) -> np.ndarray:
    selected = np.asarray(columns, dtype=int)
    standardised = (
        matrix[:, selected].astype(np.float32)
        - discovery_statistics["mean"][selected]
    ) / discovery_statistics["scale"][selected]
    return standardised.mean(axis=1)


def matched_random_panel(
    candidate_columns: Sequence[int],
    primary_columns: Sequence[int],
    layers: Sequence[int],
    latent_dim: int,
    rng: np.random.Generator,
) -> List[int]:
    primary_set = set(primary_columns)
    requested = Counter(column_to_feature(column, layers, latent_dim)[0] for column in primary_columns)
    by_layer: Dict[int, List[int]] = defaultdict(list)
    for column in candidate_columns:
        if column in primary_set:
            continue
        layer, _ = column_to_feature(column, layers, latent_dim)
        by_layer[layer].append(int(column))
    selected = []
    for layer, count in sorted(requested.items()):
        pool = np.asarray(by_layer[layer], dtype=int)
        if len(pool) < count:
            raise ValueError(f"Not enough layer-{layer} candidates for a matched random panel")
        selected.extend(int(value) for value in rng.choice(pool, size=count, replace=False))
    return selected


def build_panels(
    ranked_columns: Sequence[int],
    layers: Sequence[int],
    latent_dim: int,
    panel_sizes: Sequence[int],
    primary_size: int,
    random_panels: int,
    seed: int,
) -> List[Dict[str, Any]]:
    panels = []
    for size in sorted(set(panel_sizes)):
        name = f"top_{size}" + ("_primary" if size == primary_size else "")
        panels.append(
            {
                "name": name,
                "kind": "balanced_activation_ranked_prefix",
                "columns": list(ranked_columns[:size]),
            }
        )
    primary = list(ranked_columns[:primary_size])
    rng = np.random.default_rng(seed)
    for index in range(random_panels):
        panels.append(
            {
                "name": f"random_matched_{index + 1:02d}",
                "kind": "layer_count_matched_balanced_candidate_control",
                "columns": matched_random_panel(
                    ranked_columns,
                    primary,
                    layers,
                    latent_dim,
                    rng,
                ),
            }
        )
    return panels


def serialise_panel(panel: Dict[str, Any], layers: Sequence[int], latent_dim: int) -> Dict[str, Any]:
    features = [column_to_feature(column, layers, latent_dim) for column in panel["columns"]]
    return {
        "name": panel["name"],
        "kind": panel["kind"],
        "feature_count": len(features),
        "features": [
            {"layer": layer, "feature": feature, "key": feature_key((layer, feature))}
            for layer, feature in features
        ],
        "layer_counts": dict(Counter(layer for layer, _ in features)),
    }


def evaluate_causal_panel(
    model,
    tokenizer,
    saes,
    panel: Sequence[Feature],
    cases: Sequence[Dict[str, Any]],
    verbose: bool,
) -> List[Dict[str, Any]]:
    features = feature_dict(panel)
    layers = sorted(features)
    rows = []
    for case in cases:
        carry_baseline = case["carry_baseline"]
        no_carry_baseline = case["no_carry_baseline"]
        carry_token = carry_baseline["correct_token"]
        no_carry_token = no_carry_baseline["correct_token"]
        with suppress_output(not verbose):
            carry_result = run_inhibition_intervention(
                model,
                tokenizer,
                case["target_prompt"],
                layers,
                saes,
                features,
                [carry_token, no_carry_token],
                position_spec="last",
            )
            control_result = run_inhibition_intervention(
                model,
                tokenizer,
                case["source_prompt"],
                layers,
                saes,
                features,
                [no_carry_token, carry_token],
                position_spec="last",
            )

        carry = condition_from_result(
            carry_result,
            carry_token,
            no_carry_token,
            int(carry_baseline["correct_id"]),
            int(carry_baseline["contrast_id"]),
        )
        control = condition_from_result(
            control_result,
            no_carry_token,
            carry_token,
            int(no_carry_baseline["correct_id"]),
            int(no_carry_baseline["contrast_id"]),
        )
        carry_delta = float(carry["gap"] - carry_baseline["gap"])
        control_delta = float(control["gap"] - no_carry_baseline["gap"])
        rows.append(
            {
                "case_key": case["case_key"],
                "carry_output_digit": case["correct_digit"],
                "no_carry_output_digit": case["dropped_carry_digit"],
                "carry_delta": carry_delta,
                "no_carry_control_delta": control_delta,
                "paired_difference": carry_delta - control_delta,
                "carry_top_transferred": bool(carry["top_is_second"]),
                "control_top_transferred": bool(control["top_is_second"]),
            }
        )
    return rows


def summarise_causal_rows(rows: Sequence[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    carry = np.asarray([row["carry_delta"] for row in rows], dtype=float)
    control = np.asarray([row["no_carry_control_delta"] for row in rows], dtype=float)
    paired = carry - control
    return {
        "eligible_pairs": len(rows),
        "mean_carry_target_delta": float(carry.mean()),
        "bootstrap_95_ci_mean_carry_target_delta": list(bootstrap_mean_ci(carry, seed)),
        "mean_no_carry_control_delta": float(control.mean()),
        "bootstrap_95_ci_mean_no_carry_control_delta": list(
            bootstrap_mean_ci(control, seed + 1)
        ),
        "mean_paired_difference": float(paired.mean()),
        "bootstrap_95_ci_mean_paired_difference": list(
            bootstrap_mean_ci(paired, seed + 2)
        ),
        "fraction_carry_more_negative_than_control": float(np.mean(carry < control)),
        "fraction_carry_delta_negative": float(np.mean(carry < 0)),
        "carry_top_prediction_transfer_fraction": float(
            np.mean([row["carry_top_transferred"] for row in rows])
        ),
        "control_top_prediction_transfer_fraction": float(
            np.mean([row["control_top_transferred"] for row in rows])
        ),
    }


def protocol_signature(
    args: argparse.Namespace,
    excluded_keys: Iterable[str],
) -> Dict[str, Any]:
    return {
        "protocol_version": 1,
        "model_config": str(args.model_config),
        "sae_config": str(args.sae_config),
        "candidate_pairs": args.candidate_pairs,
        "discovery_pairs": args.discovery_pairs,
        "confirmation_pairs": args.confirmation_pairs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "panel_sizes": list(args.panel_sizes),
        "primary_panel_size": args.primary_panel_size,
        "random_panels": args.random_panels,
        "minimum_active_fraction": args.minimum_active_fraction,
        "minimum_positive_stratum_fraction": args.minimum_positive_stratum_fraction,
        "excluded_case_count": len(set(excluded_keys)),
        "excluded_case_hash": hash_strings(excluded_keys),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Output-digit-balanced carry localisation and frozen causal confirmation"
    )
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--sae-config", default="configs/sae_math_topk256_config.yaml")
    parser.add_argument("--candidate-pairs", type=int, default=149)
    parser.add_argument("--discovery-pairs", type=int, default=32)
    parser.add_argument("--confirmation-pairs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=4787)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--panel-sizes", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    parser.add_argument("--primary-panel-size", type=int, default=10)
    parser.add_argument("--random-panels", type=int, default=5)
    parser.add_argument("--minimum-active-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-positive-stratum-fraction", type=float, default=0.60)
    parser.add_argument("--exclude-json", nargs="*", default=[])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/math_carry_localization/math_topk256_balanced_carry_localization.json",
    )
    parser.add_argument(
        "--activation-cache",
        default="outputs/math_carry_localization/math_topk256_balanced_carry_activations.npz",
    )
    args = parser.parse_args()

    if args.primary_panel_size not in args.panel_sizes:
        raise ValueError("--primary-panel-size must appear in --panel-sizes")
    if max(args.panel_sizes) < args.primary_panel_size:
        raise ValueError("Panel sizes do not include the primary size")

    repo_root = get_repo_root()
    output_path = resolve_path(args.output, repo_root)
    cache_path = resolve_path(args.activation_cache, repo_root)
    exclude_paths = [resolve_path(path, repo_root) for path in args.exclude_json]
    excluded_keys, exclusion_sources = load_excluded_case_keys(exclude_paths)
    signature = protocol_signature(args, excluded_keys)

    if output_path.exists() and not args.overwrite:
        with output_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("protocol_signature") != signature:
            raise ValueError(
                f"Existing output {output_path} uses a different protocol; use --overwrite or another path"
            )
        if payload.get("status") == "complete":
            print(f"Completed result already exists: {output_path}")
            print(json.dumps(payload["primary_result"], indent=2))
            return
        print(f"Resuming partial output: {output_path}")
    else:
        payload = {
            "status": "initialising",
            "protocol_signature": signature,
            "method": {
                "candidate_universe": "all TopK-256 SAE latents at seven final-token MLP sites",
                "discovery_estimand": (
                    "mean standardised carry-minus-no-carry activation difference within shared "
                    "predicted-tens-digit strata"
                ),
                "feature_ordering": "discovery split only",
                "confirmation_data_used_for_ranking": False,
                "primary_panel": f"top_{args.primary_panel_size}_primary",
                "primary_success_rule": (
                    "positive held-out conditional activation CI; negative carry-target inhibition; "
                    "and a carry-minus-control causal CI wholly below zero"
                ),
                "causal_intervention": "final-token error-preserving inhibition",
                "scope": (
                    "Tests a compact carry-associated panel in the existing final-token MLP SAE basis; "
                    "it does not test attention or residual-stream dictionaries."
                ),
            },
            "exclusions": {
                "sources": exclusion_sources,
                "excluded_case_count": len(excluded_keys),
                "excluded_case_keys": sorted(excluded_keys),
            },
            "causal_confirmation": {"panels": []},
        }
        checkpoint_payload(payload, output_path)

    started = time.time()
    print("Loading model and selected mathematics TopK-256 SAEs...")
    model, tokenizer, _ = load_model_and_tokenizer(repo_root / args.model_config)
    saes = load_domain_saes(model, resolve_path(args.sae_config, repo_root))

    if cache_path.exists() and not args.overwrite and "split" in payload:
        print(f"Loading activation cache: {cache_path}")
        records, selected_cases, latent_matrix, raw_by_layer = load_activation_cache(
            cache_path, signature
        )
        discovery_keys = set(payload["split"]["discovery_case_keys"])
        confirmation_keys = set(payload["split"]["confirmation_case_keys"])
    else:
        generated = generated_math_cases(args.candidate_pairs, repo_root / "data/addition_data.csv", args.seed)
        fresh = [case for case in generated if arithmetic_case_key(case) not in excluded_keys]
        print(
            f"Generated {len(generated)} candidate pairs; {len(fresh)} remain after "
            f"excluding {len(excluded_keys)} previously inspected keys"
        )
        prepared = screen_baselines(model, tokenizer, fresh, args.batch_size)
        eligible = [case for case in prepared if case["eligible"]]
        discovery_cases, confirmation_cases, split_balance = choose_balanced_split(
            eligible,
            args.discovery_pairs,
            args.confirmation_pairs,
            args.seed + 100,
        )
        selected_cases = discovery_cases + confirmation_cases
        discovery_keys = {case["case_key"] for case in discovery_cases}
        confirmation_keys = {case["case_key"] for case in confirmation_cases}
        payload["baseline_screen"] = {
            "fresh_candidate_pairs": len(fresh),
            "eligible_pairs": len(eligible),
            "ineligible_pairs": len(fresh) - len(eligible),
            "cases": [public_case(case) for case in prepared],
        }
        payload["split"] = {
            "split_frozen_before_feature_scoring": True,
            "selection_used_feature_activations": False,
            "discovery_case_keys": sorted(discovery_keys),
            "confirmation_case_keys": sorted(confirmation_keys),
            "balance": split_balance,
        }
        payload["status"] = "activation_capture"
        checkpoint_payload(payload, output_path)

        records, latent_matrix, raw_by_layer = collect_activation_dataset(
            model,
            tokenizer,
            saes,
            selected_cases,
            args.batch_size,
        )
        save_activation_cache(
            cache_path,
            signature,
            records,
            selected_cases,
            latent_matrix,
            raw_by_layer,
        )
        print(f"Saved activation cache: {cache_path}")

    discovery_indices = indices_for_pairs(records, discovery_keys)
    confirmation_indices = indices_for_pairs(records, confirmation_keys)
    discovery_records = [records[int(index)] for index in discovery_indices]
    confirmation_records = [records[int(index)] for index in confirmation_indices]
    discovery_latent = latent_matrix[discovery_indices]
    confirmation_latent = latent_matrix[confirmation_indices]

    print("Fitting output-digit-conditioned raw MLP directions...")
    raw_results = []
    for layer in LAYERS:
        fit = fit_conditional_direction(raw_by_layer[layer][discovery_indices], discovery_records)
        discovery_scores = fit["training_scores"]
        confirmation_scores = apply_conditional_direction(
            fit,
            raw_by_layer[layer][confirmation_indices],
            confirmation_records,
        )
        raw_results.append(
            {
                "layer": layer,
                "probe": "standardised mean-difference direction after output-digit centring",
                "discovery": conditional_score_summary(
                    discovery_scores, discovery_records, args.seed + layer
                ),
                "confirmation": conditional_score_summary(
                    confirmation_scores, confirmation_records, args.seed + 1000 + layer
                ),
            }
        )
        print(
            f"  layer {layer}: confirmation conditioned AUC "
            f"{raw_results[-1]['confirmation']['output_digit_conditioned_auc']:.3f}"
        )
    payload["raw_mlp_localisation"] = raw_results

    print("Ranking all SAE features on balanced discovery activations...")
    ranked_columns, discovery_statistics, ranking_records = rank_carry_features(
        discovery_latent,
        discovery_records,
        LAYERS,
        args.minimum_active_fraction,
        args.minimum_positive_stratum_fraction,
    )
    latent_dim = int(discovery_statistics["latent_dim"].item())
    payload["sae_feature_discovery"] = {
        "total_feature_count": int(discovery_latent.shape[1]),
        "candidate_count_after_fixed_filters": len(ranked_columns),
        "candidate_layer_counts": dict(
            Counter(column_to_feature(column, LAYERS, latent_dim)[0] for column in ranked_columns)
        ),
        "minimum_active_fraction": args.minimum_active_fraction,
        "minimum_positive_stratum_fraction": args.minimum_positive_stratum_fraction,
        "required_positive_strata": int(discovery_statistics["required_positive_strata"].item()),
        "top_200_ranking": ranking_records,
        "frozen_top_20": [
            {
                "layer": column_to_feature(column, LAYERS, latent_dim)[0],
                "feature": column_to_feature(column, LAYERS, latent_dim)[1],
                "key": feature_key(column_to_feature(column, LAYERS, latent_dim)),
            }
            for column in ranked_columns[:20]
        ],
        "confirmation_was_used_for_ranking": False,
    }

    panels = build_panels(
        ranked_columns,
        LAYERS,
        latent_dim,
        args.panel_sizes,
        args.primary_panel_size,
        args.random_panels,
        args.seed + 2000,
    )
    panel_activation_results: Dict[str, Dict[str, Any]] = {}
    for index, panel in enumerate(panels, start=1):
        discovery_scores = panel_scores(
            discovery_latent, panel["columns"], discovery_statistics
        )
        confirmation_scores = panel_scores(
            confirmation_latent, panel["columns"], discovery_statistics
        )
        panel_activation_results[panel["name"]] = {
            "discovery": conditional_score_summary(
                discovery_scores, discovery_records, args.seed + 3000 + index
            ),
            "confirmation": conditional_score_summary(
                confirmation_scores, confirmation_records, args.seed + 4000 + index
            ),
        }
    payload["panel_activation_validation"] = panel_activation_results
    payload["status"] = "causal_confirmation"
    checkpoint_payload(payload, output_path)

    confirmation_case_map = {case["case_key"]: case for case in selected_cases}
    confirmation_cases = [confirmation_case_map[key] for key in sorted(confirmation_keys)]
    completed_panels = {
        panel["name"] for panel in payload["causal_confirmation"].get("panels", [])
    }
    for index, panel in enumerate(panels, start=1):
        if panel["name"] in completed_panels:
            print(f"[causal panel {index:02d}/{len(panels)}] {panel['name']} already complete")
            continue
        features = [column_to_feature(column, LAYERS, latent_dim) for column in panel["columns"]]
        print(
            f"[causal panel {index:02d}/{len(panels)}] {panel['name']} "
            f"({len(features)} features on {len(confirmation_cases)} pairs)"
        )
        rows = evaluate_causal_panel(
            model,
            tokenizer,
            saes,
            features,
            confirmation_cases,
            args.verbose,
        )
        result = serialise_panel(panel, LAYERS, latent_dim)
        result["activation_validation"] = panel_activation_results[panel["name"]]
        result["causal_summary"] = summarise_causal_rows(
            rows, args.seed + 5000 + index * 10
        )
        result["case_effects"] = rows
        payload["causal_confirmation"]["panels"].append(result)
        checkpoint_payload(payload, output_path)

    primary_name = f"top_{args.primary_panel_size}_primary"
    primary = next(
        panel
        for panel in payload["causal_confirmation"]["panels"]
        if panel["name"] == primary_name
    )
    activation_summary = primary["activation_validation"]["confirmation"]
    causal_summary = primary["causal_summary"]
    activation_ci = activation_summary["bootstrap_95_ci_mean_within_digit_difference"]
    paired_ci = causal_summary["bootstrap_95_ci_mean_paired_difference"]
    success = bool(
        activation_ci[0] > 0
        and causal_summary["mean_carry_target_delta"] < 0
        and paired_ci[1] < 0
    )
    random_effects = [
        panel["causal_summary"]["mean_paired_difference"]
        for panel in payload["causal_confirmation"]["panels"]
        if panel["kind"] == "layer_count_matched_balanced_candidate_control"
    ]
    payload["primary_result"] = {
        "panel": primary_name,
        "supports_compact_carry_selectivity_under_predeclared_rule": success,
        "activation_confirmation": activation_summary,
        "causal_confirmation": causal_summary,
        "random_control_mean_paired_differences": random_effects,
        "interpretation_gate": (
            "Both activation and causal criteria passed; require an independent replication."
            if success
            else "The existing final-token MLP SAE basis did not isolate a compact carry-selective panel."
        ),
    }
    payload["status"] = "complete"
    payload["runtime_seconds"] = time.time() - started
    checkpoint_payload(payload, output_path)

    print("\nPrimary balanced carry-localisation result")
    print(f"  panel: {primary_name}")
    print(
        "  activation carry-minus-no-carry: "
        f"{activation_summary['mean_within_digit_carry_minus_no_carry']:+.4f} "
        f"(95% CI [{activation_ci[0]:+.4f}, {activation_ci[1]:+.4f}])"
    )
    print(
        "  causal carry-minus-control: "
        f"{causal_summary['mean_paired_difference']:+.4f} "
        f"(95% CI [{paired_ci[0]:+.4f}, {paired_ci[1]:+.4f}])"
    )
    print(f"  predeclared joint success rule met: {success}")
    print(f"Saved result to {output_path}")
    print(f"Activation cache: {cache_path}")


if __name__ == "__main__":
    main()

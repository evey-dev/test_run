"""Localise capital-relation features without attribution-graph preselection.

The experiment ranks every latent in the selected capitals SAE using paired
capital-versus-country prompts on discovery countries. A frozen panel is then
validated by activation and final-token, error-preserving inhibition on
disjoint confirmation countries. Prompt eligibility and split size are fixed
before any SAE feature activation is inspected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from src.capitals_relation_feature_screen import (
    PROMPT_PAIRS,
    continuation_token,
    evaluate_panel,
    read_capitals,
    summarise_rows,
)
from src.data_utils import get_repo_root, resolve_path
from src.heldout_validation import LAYERS, bootstrap_mean_ci, load_domain_saes
from src.math_carry_balanced_localization import (
    batched,
    build_panels,
    checkpoint_payload,
    column_to_feature,
    feature_key,
    panel_scores,
    serialise_panel,
)
from src.model_loader import load_model_and_tokenizer


Feature = Tuple[int, int]
PROTOCOL_VERSION = 1


def hash_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(values)):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_intervened_countries(paths: Sequence[Path]) -> Tuple[set[str], List[Dict[str, Any]]]:
    """Read only countries used for feature interventions, not baseline rejects."""
    countries: set[str] = set()
    sources = []
    for path in paths:
        if not path.exists():
            sources.append({"path": str(path), "exists": False, "country_count": 0})
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        selection = payload.get("case_selection", {})
        selected_rows = []
        for key in ("discovery_cases", "confirmation_cases"):
            value = selection.get(key, [])
            if isinstance(value, list):
                selected_rows.extend(row for row in value if isinstance(row, dict))
        source_countries = {
            str(row["country"]).strip()
            for row in selected_rows
            if row.get("country")
        }
        countries.update(source_countries)
        sources.append(
            {
                "path": str(path),
                "exists": True,
                "country_count": len(source_countries),
                "countries": sorted(source_countries),
            }
        )
    return countries, sources


def candidate_cases(
    rows: Sequence[Dict[str, str]],
    seed: int,
    excluded_countries: Iterable[str],
    sae_corpus_prompts: set[str],
    cities_per_country: int,
) -> List[Dict[str, Any]]:
    excluded = {value.casefold() for value in excluded_countries}
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["Type"].strip().casefold() != "country":
            continue
        country = row["Location"].strip()
        if country.casefold() not in excluded:
            grouped[country].append(row)

    rng = np.random.default_rng(seed)
    countries = sorted(grouped)
    rng.shuffle(countries)
    variants_by_country: Dict[str, List[Dict[str, Any]]] = {}
    for country in countries:
        country_rows = list(grouped[country])
        rng.shuffle(country_rows)
        variants = []
        for source in country_rows[:cities_per_country]:
            city = source["DistractorAnswer"].strip()
            for template_name, capital_template, country_template in PROMPT_PAIRS:
                capital_prompt = capital_template.format(city=city)
                country_prompt = country_template.format(city=city)
                variants.append(
                    {
                        "case_key": country,
                        "country": country,
                        "capital": source["Answer"].strip(),
                        "city": city,
                        "template": template_name,
                        "capital_prompt": capital_prompt,
                        "country_prompt": country_prompt,
                        "capital_prompt_absent_from_sae_corpus": (
                            capital_prompt not in sae_corpus_prompts
                        ),
                        "country_prompt_absent_from_sae_corpus": (
                            country_prompt not in sae_corpus_prompts
                        ),
                    }
                )
        variants_by_country[country] = variants

    # Interleave countries so baseline screening does not spend whole batches on
    # alternate phrasings for a country that may already have an eligible pair.
    cases = []
    maximum_variants = max(
        (len(values) for values in variants_by_country.values()), default=0
    )
    for variant_index in range(maximum_variants):
        for country in countries:
            variants = variants_by_country[country]
            if variant_index < len(variants):
                cases.append(variants[variant_index])
    return cases


def condition_from_batch(
    logits: torch.Tensor,
    probabilities: torch.Tensor,
    top_id: int,
    top_token: str,
    first_id: int,
    second_id: int,
) -> Dict[str, Any]:
    first_logit = float(logits[first_id].item())
    second_logit = float(logits[second_id].item())
    return {
        "first": {
            "logit": first_logit,
            "probability": float(probabilities[first_id].item()),
        },
        "second": {
            "logit": second_logit,
            "probability": float(probabilities[second_id].item()),
        },
        "gap": first_logit - second_logit,
        "top_token": top_token,
        "top_token_id": int(top_id),
        "top_is_first": bool(top_id == first_id),
        "top_is_second": bool(top_id == second_id),
    }


def screen_candidate_cases(
    model,
    tokenizer,
    cases: Sequence[Dict[str, Any]],
    desired_eligible: int,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Select the first baseline-correct prompt variant for each country."""
    selected: Dict[str, Dict[str, Any]] = {}
    screened: List[Dict[str, Any]] = []
    cursor = 0
    batch_index = 0
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    while cursor < len(cases) and len(selected) < desired_eligible:
        candidate_batch = []
        while cursor < len(cases) and len(candidate_batch) < batch_size:
            case = cases[cursor]
            cursor += 1
            if case["country"] in selected:
                continue
            row = dict(case)
            try:
                capital_id, capital_token = continuation_token(tokenizer, row["capital"])
                country_id, country_token = continuation_token(tokenizer, row["country"])
            except ValueError as exc:
                row.update(
                    {
                        "eligible": False,
                        "selected_for_split_pool": False,
                        "ineligible_reason": str(exc),
                    }
                )
                screened.append(row)
                continue
            if capital_id == country_id:
                row.update(
                    {
                        "capital_id": capital_id,
                        "country_id": country_id,
                        "capital_token": capital_token,
                        "country_token": country_token,
                        "eligible": False,
                        "selected_for_split_pool": False,
                        "ineligible_reason": "capital and country share first token",
                    }
                )
                screened.append(row)
                continue
            row.update(
                {
                    "capital_id": capital_id,
                    "country_id": country_id,
                    "capital_token": capital_token,
                    "country_token": country_token,
                }
            )
            candidate_batch.append(row)

        if not candidate_batch:
            continue
        batch_index += 1
        prompts = []
        for row in candidate_batch:
            prompts.extend([row["capital_prompt"], row["country_prompt"]])
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
            probabilities = torch.softmax(logits.float(), dim=-1)
            top_ids = logits.argmax(dim=-1)

        for index, row in enumerate(candidate_batch):
            capital_row = 2 * index
            country_row = capital_row + 1
            capital_top_id = int(top_ids[capital_row].item())
            country_top_id = int(top_ids[country_row].item())
            capital_clean = condition_from_batch(
                logits[capital_row],
                probabilities[capital_row],
                capital_top_id,
                tokenizer.decode([capital_top_id]),
                int(row["capital_id"]),
                int(row["country_id"]),
            )
            country_clean = condition_from_batch(
                logits[country_row],
                probabilities[country_row],
                country_top_id,
                tokenizer.decode([country_top_id]),
                int(row["capital_id"]),
                int(row["country_id"]),
            )
            reasons = []
            if not capital_clean["top_is_first"]:
                reasons.append("capital prompt did not predict capital prefix")
            if not country_clean["top_is_second"]:
                reasons.append("inverse prompt did not predict country prefix")
            eligible = not reasons
            selected_now = bool(
                eligible
                and row["country"] not in selected
                and len(selected) < desired_eligible
            )
            row.update(
                {
                    "capital_clean": capital_clean,
                    "country_clean": country_clean,
                    "eligible": eligible,
                    "selected_for_split_pool": selected_now,
                    "ineligible_reason": "; ".join(reasons) if reasons else None,
                }
            )
            screened.append(row)
            if selected_now:
                selected[row["country"]] = row

        print(
            f"[baseline batch {batch_index:02d}] screened {len(candidate_batch)} variants; "
            f"selected {len(selected)}/{desired_eligible} countries"
        )

    return list(selected.values()), screened


def choose_split(
    eligible: Sequence[Dict[str, Any]],
    desired_per_split: int,
    minimum_per_split: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    actual = min(desired_per_split, len(eligible) // 2)
    if actual < minimum_per_split:
        raise ValueError(
            f"Only {len(eligible)} baseline-eligible countries were available; at least "
            f"{2 * minimum_per_split} are required"
        )
    order = np.random.default_rng(seed).permutation(len(eligible))[: 2 * actual]
    discovery = [eligible[int(index)] for index in order[:actual]]
    confirmation = [eligible[int(index)] for index in order[actual:]]
    return discovery, confirmation, {
        "desired_cases_per_split": desired_per_split,
        "minimum_cases_per_split": minimum_per_split,
        "actual_cases_per_split": actual,
        "fallback_used": actual < desired_per_split,
        "fallback_depended_only_on_baseline_eligibility": True,
        "discovery_template_counts": dict(Counter(row["template"] for row in discovery)),
        "confirmation_template_counts": dict(Counter(row["template"] for row in confirmation)),
    }


def public_case(case: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "case_key",
        "country",
        "capital",
        "city",
        "template",
        "capital_prompt",
        "country_prompt",
        "capital_prompt_absent_from_sae_corpus",
        "country_prompt_absent_from_sae_corpus",
        "capital_id",
        "country_id",
        "capital_token",
        "country_token",
        "capital_clean",
        "country_clean",
        "eligible",
        "selected_for_split_pool",
        "ineligible_reason",
    ]
    return {key: case[key] for key in keys if key in case}


def observation_records(cases: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for case in cases:
        records.extend(
            [
                {
                    "case_key": case["case_key"],
                    "country": case["country"],
                    "condition": "capital",
                    "relation_label": 1,
                    "template": case["template"],
                    "prompt": case["capital_prompt"],
                },
                {
                    "case_key": case["case_key"],
                    "country": case["country"],
                    "condition": "inverse_country",
                    "relation_label": 0,
                    "template": case["template"],
                    "prompt": case["country_prompt"],
                },
            ]
        )
    return records


def collect_activation_dataset(
    model,
    tokenizer,
    saes,
    cases: Sequence[Dict[str, Any]],
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    records = observation_records(cases)
    chunks: Dict[int, List[np.ndarray]] = {layer: [] for layer in LAYERS}
    captured: Dict[int, np.ndarray] = {}
    hooks = []

    def make_hook(layer: int):
        sae, scaling_factor = saes[layer]

        def hook_fn(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            raw = tensor[:, -1, :]
            with torch.no_grad():
                latent = sae.encode(raw / scaling_factor)
            captured[layer] = latent.detach().float().cpu().numpy().astype(np.float16)

        return hook_fn

    for layer in LAYERS:
        hooks.append(model.model.layers[layer].mlp.register_forward_hook(make_hook(layer)))

    tokenizer.padding_side = "left"
    try:
        for batch_index, batch in enumerate(batched(records, batch_size), start=1):
            captured.clear()
            inputs = tokenizer(
                [row["prompt"] for row in batch],
                return_tensors="pt",
                padding=True,
            ).to(model.device)
            with torch.no_grad():
                model(**inputs)
            missing = set(LAYERS).difference(captured)
            if missing:
                raise RuntimeError(f"MLP hooks did not fire for layers {sorted(missing)}")
            for layer in LAYERS:
                chunks[layer].append(captured[layer])
            print(f"[activation batch {batch_index:02d}] captured {len(batch)} prompts")
    finally:
        for hook in hooks:
            hook.remove()

    latent = np.concatenate(
        [np.concatenate(chunks[layer], axis=0) for layer in LAYERS],
        axis=1,
    )
    if latent.shape[0] != len(records):
        raise RuntimeError("Activation rows do not align with observation records")
    return records, latent


def save_activation_cache(
    path: Path,
    signature: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    selected_cases: Sequence[Dict[str, Any]],
    latent: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            signature=np.asarray(json.dumps(signature, sort_keys=True)),
            records=np.asarray(json.dumps(list(records))),
            selected_cases=np.asarray(json.dumps([public_case(case) for case in selected_cases])),
            latent=latent.astype(np.float16),
        )
    temporary.replace(path)


def load_activation_cache(
    path: Path,
    signature: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        stored_signature = json.loads(str(payload["signature"].item()))
        if stored_signature != signature:
            raise ValueError(
                f"Activation cache {path} uses a different protocol; choose another path"
            )
        records = json.loads(str(payload["records"].item()))
        selected_cases = json.loads(str(payload["selected_cases"].item()))
        latent = payload["latent"].astype(np.float32)
    return records, selected_cases, latent


def indices_for_countries(records: Sequence[Dict[str, Any]], countries: Iterable[str]) -> np.ndarray:
    selected = set(countries)
    return np.asarray(
        [index for index, row in enumerate(records) if row["country"] in selected],
        dtype=int,
    )


def paired_differences(
    values: np.ndarray,
    records: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, List[str]]:
    grouped: Dict[str, Dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(records):
        grouped[row["country"]][row["condition"]] = index
    countries = sorted(grouped)
    differences = []
    for country in countries:
        pair = grouped[country]
        if set(pair) != {"capital", "inverse_country"}:
            raise ValueError(f"Country {country!r} does not have one prompt per condition")
        differences.append(values[pair["capital"]] - values[pair["inverse_country"]])
    return np.asarray(differences), countries


def paired_score_summary(
    scores: np.ndarray,
    records: Sequence[Dict[str, Any]],
    seed: int,
) -> Dict[str, Any]:
    differences, countries = paired_differences(np.asarray(scores), records)
    return {
        "country_count": len(countries),
        "mean_capital_minus_inverse_score": float(differences.mean()),
        "bootstrap_95_ci_mean_capital_minus_inverse_score": list(
            bootstrap_mean_ci(differences, seed)
        ),
        "paired_relation_accuracy": float(
            np.mean(differences > 0) + 0.5 * np.mean(differences == 0)
        ),
        "fraction_capital_score_greater": float(np.mean(differences > 0)),
        "median_capital_minus_inverse_score": float(np.median(differences)),
    }


def rank_relation_features(
    discovery_matrix: np.ndarray,
    discovery_records: Sequence[Dict[str, Any]],
    layers: Sequence[int],
    minimum_active_fraction: float,
    minimum_positive_pair_fraction: float,
) -> Tuple[List[int], Dict[str, np.ndarray], List[Dict[str, Any]]]:
    matrix = discovery_matrix.astype(np.float32)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    safe_scale = scale.copy()
    safe_scale[safe_scale < 1e-5] = 1.0
    standardised = (matrix - mean) / safe_scale
    differences, countries = paired_differences(standardised, discovery_records)
    effect = differences.mean(axis=0)
    positive_pairs = (differences > 0).sum(axis=0)
    required_positive = max(
        1, math.ceil(minimum_positive_pair_fraction * len(countries))
    )
    labels = np.asarray([row["relation_label"] for row in discovery_records], dtype=int)
    capital_active = (matrix[labels == 1] > 1e-6).mean(axis=0)
    inverse_active = (matrix[labels == 0] > 1e-6).mean(axis=0)
    candidate_mask = (
        (effect > 0)
        & (capital_active >= minimum_active_fraction)
        & (positive_pairs >= required_positive)
        & (scale >= 1e-5)
    )
    candidates = np.flatnonzero(candidate_mask)
    if len(candidates) < 20:
        raise ValueError(
            f"Only {len(candidates)} relation-associated candidates passed the fixed filters"
        )
    ordering = candidates[
        np.lexsort(
            (
                candidates,
                inverse_active[candidates],
                -capital_active[candidates],
                -positive_pairs[candidates],
                -effect[candidates],
            )
        )
    ]

    latent_dim = matrix.shape[1] // len(layers)
    if latent_dim * len(layers) != matrix.shape[1]:
        raise ValueError("Latent matrix width is not divisible by the layer count")

    ranking = []
    for rank, column in enumerate(ordering[:200], start=1):
        layer, feature = column_to_feature(int(column), layers, latent_dim)
        ranking.append(
            {
                "rank": rank,
                "key": feature_key((layer, feature)),
                "layer": layer,
                "feature": feature,
                "column": int(column),
                "paired_standardised_effect": float(effect[column]),
                "positive_discovery_pairs": int(positive_pairs[column]),
                "total_discovery_pairs": len(countries),
                "capital_active_fraction": float(capital_active[column]),
                "inverse_country_active_fraction": float(inverse_active[column]),
            }
        )
    statistics = {
        "mean": mean,
        "scale": safe_scale,
        "effect": effect,
        "positive_pairs": positive_pairs,
        "capital_active": capital_active,
        "inverse_active": inverse_active,
        "latent_dim": np.asarray(latent_dim),
        "required_positive_pairs": np.asarray(required_positive),
    }
    return [int(column) for column in ordering], statistics, ranking


def protocol_signature(
    args: argparse.Namespace,
    excluded_countries: Iterable[str],
) -> Dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model_config": str(args.model_config),
        "sae_config": str(args.sae_config),
        "capitals_csv": str(args.capitals_csv),
        "sae_corpus_csv": str(args.sae_corpus_csv),
        "desired_cases_per_split": args.desired_cases_per_split,
        "minimum_cases_per_split": args.minimum_cases_per_split,
        "cities_per_country": args.cities_per_country,
        "seed": args.seed,
        "candidate_batch_size": args.candidate_batch_size,
        "activation_batch_size": args.activation_batch_size,
        "panel_sizes": list(args.panel_sizes),
        "primary_panel_size": args.primary_panel_size,
        "random_panels": args.random_panels,
        "minimum_active_fraction": args.minimum_active_fraction,
        "minimum_positive_pair_fraction": args.minimum_positive_pair_fraction,
        "excluded_country_count": len(set(excluded_countries)),
        "excluded_country_hash": hash_strings(excluded_countries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="All-latent capital-relation localisation and frozen confirmation"
    )
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--sae-config", required=True)
    parser.add_argument("--capitals-csv", default="data/capitals_data.csv")
    parser.add_argument("--sae-corpus-csv", default="data/capitals_large_10000.csv")
    parser.add_argument("--exclude-json", nargs="*", default=[])
    parser.add_argument("--exclude-country", action="append", default=[])
    parser.add_argument("--desired-cases-per-split", type=int, default=24)
    parser.add_argument("--minimum-cases-per-split", type=int, default=16)
    parser.add_argument("--cities-per-country", type=int, default=6)
    parser.add_argument("--seed", type=int, default=8787)
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    parser.add_argument("--activation-batch-size", type=int, default=8)
    parser.add_argument("--panel-sizes", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    parser.add_argument("--primary-panel-size", type=int, default=3)
    parser.add_argument("--random-panels", type=int, default=5)
    parser.add_argument("--minimum-active-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-positive-pair-fraction", type=float, default=0.60)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output",
        default=(
            "outputs/capitals_balanced_localization/"
            "capitals_10k_balanced_relation_localization.json"
        ),
    )
    parser.add_argument(
        "--activation-cache",
        default=(
            "outputs/capitals_balanced_localization/"
            "capitals_10k_balanced_relation_activations.npz"
        ),
    )
    args = parser.parse_args()

    if args.primary_panel_size not in args.panel_sizes:
        raise ValueError("--primary-panel-size must appear in --panel-sizes")
    if args.minimum_cases_per_split > args.desired_cases_per_split:
        raise ValueError("minimum cases cannot exceed desired cases")

    repo_root = get_repo_root()
    output_path = resolve_path(args.output, repo_root)
    cache_path = resolve_path(args.activation_cache, repo_root)
    exclude_paths = [resolve_path(path, repo_root) for path in args.exclude_json]
    prior_countries, exclusion_sources = load_intervened_countries(exclude_paths)
    excluded_countries = prior_countries | {
        value.strip() for value in args.exclude_country if value.strip()
    }
    signature = protocol_signature(args, excluded_countries)

    if output_path.exists() and not args.overwrite:
        with output_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("protocol_signature") != signature:
            raise ValueError(
                f"Existing output {output_path} uses a different protocol; use another path"
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
                "candidate_universe": (
                    "all TopK-128 SAE latents at seven final-token MLP sites"
                ),
                "discovery_estimand": (
                    "mean standardised within-country capital-prompt minus "
                    "inverse-country-prompt activation"
                ),
                "feature_ordering": "discovery countries only",
                "confirmation_data_used_for_ranking": False,
                "primary_panel": f"top_{args.primary_panel_size}_primary",
                "primary_success_rule": (
                    "positive held-out paired activation CI, negative capital-prompt "
                    "inhibition with a negative CI, and a relation-specific causal CI "
                    "wholly below zero"
                ),
                "causal_intervention": "final-token error-preserving inhibition",
                "categorical_country_flip_required": False,
            },
            "exclusions": {
                "sources": exclusion_sources,
                "excluded_country_count": len(excluded_countries),
                "excluded_countries": sorted(excluded_countries),
            },
            "causal_confirmation": {"panels": []},
        }
        checkpoint_payload(payload, output_path)

    started = time.time()
    print("Loading model and capitals TopK-128 SAEs...")
    model, tokenizer, _ = load_model_and_tokenizer(
        resolve_path(args.model_config, repo_root)
    )
    saes = load_domain_saes(model, resolve_path(args.sae_config, repo_root))

    if cache_path.exists() and not args.overwrite and "split" in payload:
        print(f"Loading activation cache: {cache_path}")
        records, selected_cases, latent_matrix = load_activation_cache(
            cache_path, signature
        )
        discovery_countries = set(payload["split"]["discovery_countries"])
        confirmation_countries = set(payload["split"]["confirmation_countries"])
    else:
        capitals_path = resolve_path(args.capitals_csv, repo_root)
        corpus_path = resolve_path(args.sae_corpus_csv, repo_root)
        with corpus_path.open("r", encoding="utf-8", newline="") as handle:
            corpus_prompts = {row["sentence"] for row in csv.DictReader(handle)}
        candidates = candidate_cases(
            read_capitals(capitals_path),
            args.seed,
            excluded_countries,
            corpus_prompts,
            args.cities_per_country,
        )
        desired_total = 2 * args.desired_cases_per_split
        eligible, screened = screen_candidate_cases(
            model,
            tokenizer,
            candidates,
            desired_total,
            args.candidate_batch_size,
        )
        discovery, confirmation, split_rule = choose_split(
            eligible,
            args.desired_cases_per_split,
            args.minimum_cases_per_split,
            args.seed + 100,
        )
        selected_cases = discovery + confirmation
        discovery_countries = {row["country"] for row in discovery}
        confirmation_countries = {row["country"] for row in confirmation}
        exact_overlap = [
            row["country"]
            for row in selected_cases
            if not (
                row["capital_prompt_absent_from_sae_corpus"]
                and row["country_prompt_absent_from_sae_corpus"]
            )
        ]
        if exact_overlap:
            raise ValueError(
                "Selected prompts overlap the SAE corpus for countries: "
                + ", ".join(sorted(exact_overlap))
            )
        payload["baseline_screen"] = {
            "candidate_variant_count": len(candidates),
            "screened_variant_count": len(screened),
            "eligible_distinct_countries": len(eligible),
            "selected_cases": [public_case(row) for row in selected_cases],
            "screened_cases": [public_case(row) for row in screened],
            "all_selected_prompts_absent_from_exact_sae_corpus": True,
        }
        payload["split"] = {
            "split_frozen_before_feature_scoring": True,
            "selection_used_only_baseline_correctness": True,
            "countries_disjoint_between_splits": discovery_countries.isdisjoint(
                confirmation_countries
            ),
            "discovery_countries": sorted(discovery_countries),
            "confirmation_countries": sorted(confirmation_countries),
            "rule": split_rule,
        }
        payload["status"] = "activation_capture"
        checkpoint_payload(payload, output_path)

        records, latent_matrix = collect_activation_dataset(
            model,
            tokenizer,
            saes,
            selected_cases,
            args.activation_batch_size,
        )
        save_activation_cache(
            cache_path,
            signature,
            records,
            selected_cases,
            latent_matrix,
        )
        print(f"Saved activation cache: {cache_path}")

    discovery_indices = indices_for_countries(records, discovery_countries)
    confirmation_indices = indices_for_countries(records, confirmation_countries)
    discovery_records = [records[int(index)] for index in discovery_indices]
    confirmation_records = [records[int(index)] for index in confirmation_indices]
    discovery_latent = latent_matrix[discovery_indices]
    confirmation_latent = latent_matrix[confirmation_indices]

    print("Ranking the complete SAE latent universe on discovery countries...")
    ranked_columns, statistics, ranking = rank_relation_features(
        discovery_latent,
        discovery_records,
        LAYERS,
        args.minimum_active_fraction,
        args.minimum_positive_pair_fraction,
    )
    latent_dim = int(statistics["latent_dim"].item())
    payload["sae_feature_discovery"] = {
        "total_feature_count": int(discovery_latent.shape[1]),
        "candidate_count_after_fixed_filters": len(ranked_columns),
        "candidate_layer_counts": dict(
            Counter(
                column_to_feature(column, LAYERS, latent_dim)[0]
                for column in ranked_columns
            )
        ),
        "minimum_active_fraction": args.minimum_active_fraction,
        "minimum_positive_pair_fraction": args.minimum_positive_pair_fraction,
        "required_positive_pairs": int(statistics["required_positive_pairs"].item()),
        "top_200_ranking": ranking,
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
    activation_results: Dict[str, Dict[str, Any]] = {}
    for index, panel in enumerate(panels, start=1):
        discovery_scores = panel_scores(discovery_latent, panel["columns"], statistics)
        confirmation_scores = panel_scores(
            confirmation_latent, panel["columns"], statistics
        )
        activation_results[panel["name"]] = {
            "discovery": paired_score_summary(
                discovery_scores, discovery_records, args.seed + 3000 + index
            ),
            "confirmation": paired_score_summary(
                confirmation_scores, confirmation_records, args.seed + 4000 + index
            ),
        }
    payload["panel_activation_validation"] = activation_results
    payload["status"] = "causal_confirmation"
    checkpoint_payload(payload, output_path)

    case_map = {row["country"]: row for row in selected_cases}
    confirmation_cases = [case_map[country] for country in sorted(confirmation_countries)]
    completed = {
        panel["name"] for panel in payload["causal_confirmation"].get("panels", [])
    }
    for index, panel in enumerate(panels, start=1):
        if panel["name"] in completed:
            print(f"[causal panel {index:02d}/{len(panels)}] {panel['name']} already complete")
            continue
        features = [
            column_to_feature(column, LAYERS, latent_dim)
            for column in panel["columns"]
        ]
        print(
            f"[causal panel {index:02d}/{len(panels)}] {panel['name']} "
            f"({len(features)} features on {len(confirmation_cases)} countries)"
        )
        rows = evaluate_panel(
            model,
            tokenizer,
            saes,
            features,
            confirmation_cases,
            args.verbose,
        )
        result = serialise_panel(panel, LAYERS, latent_dim)
        result["activation_validation"] = activation_results[panel["name"]]
        result["causal_summary"] = summarise_rows(
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
    activation_ci = activation_summary[
        "bootstrap_95_ci_mean_capital_minus_inverse_score"
    ]
    capital_ci = causal_summary["bootstrap_95_ci_mean_capital_prompt_delta"]
    paired_ci = causal_summary[
        "bootstrap_95_ci_mean_relation_specific_difference"
    ]
    success = bool(
        activation_ci[0] > 0
        and causal_summary["mean_capital_prompt_delta"] < 0
        and capital_ci[1] < 0
        and causal_summary["mean_relation_specific_difference"] < 0
        and paired_ci[1] < 0
    )
    random_effects = [
        panel["causal_summary"]["mean_relation_specific_difference"]
        for panel in payload["causal_confirmation"]["panels"]
        if panel["kind"] == "layer_count_matched_balanced_candidate_control"
    ]
    payload["primary_result"] = {
        "panel": primary_name,
        "supports_compact_capital_relation_selectivity_under_predeclared_rule": success,
        "activation_confirmation": activation_summary,
        "causal_confirmation": causal_summary,
        "random_control_mean_relation_specific_differences": random_effects,
        "primary_more_negative_than_random_control_fraction": (
            float(
                np.mean(
                    causal_summary["mean_relation_specific_difference"]
                    < np.asarray(random_effects)
                )
            )
            if random_effects
            else None
        ),
        "interpretation_gate": (
            "The all-latent activation-ranked Top-3 passed held-out activation and causal criteria."
            if success
            else "The all-latent search did not confirm a compact Top-3 capital-relation panel."
        ),
    }
    payload["status"] = "complete"
    payload["runtime_seconds"] = time.time() - started
    checkpoint_payload(payload, output_path)

    print("\nPrimary balanced capital-relation result")
    print(f"  panel: {primary_name}")
    print(
        "  held-out activation relation effect: "
        f"{activation_summary['mean_capital_minus_inverse_score']:+.4f} "
        f"(95% CI [{activation_ci[0]:+.4f}, {activation_ci[1]:+.4f}])"
    )
    print(
        "  held-out causal relation specificity: "
        f"{causal_summary['mean_relation_specific_difference']:+.4f} "
        f"(95% CI [{paired_ci[0]:+.4f}, {paired_ci[1]:+.4f}])"
    )
    print(f"  predeclared joint success rule met: {success}")
    print(f"Saved result to {output_path}")
    print(f"Activation cache: {cache_path}")


if __name__ == "__main__":
    main()

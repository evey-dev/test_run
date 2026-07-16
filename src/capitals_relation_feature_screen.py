"""Discover and confirm a compact capital-relation SAE feature panel.

Candidate features come from one contrastive capital graph.  Features are
ranked on a discovery split by the effect of final-token inhibition on the
capital-minus-country logit gap.  The matched control applies the same edit to
an inverse prompt asking for the country.  Countries are disjoint between
discovery and confirmation, so entity-specific effects cannot by themselves
pass the confirmatory test.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.data_utils import get_repo_root, resolve_path
from src.heldout_validation import (
    LAYERS,
    baseline_condition,
    bootstrap_mean_ci,
    condition_from_result,
    load_domain_saes,
    suppress_output,
)
from src.intervention import run_inhibition_intervention
from src.model_loader import load_model_and_tokenizer


Feature = Tuple[int, int]
PROTOCOL_VERSION = 1

PROMPT_PAIRS = [
    (
        "country_has_capital",
        "Fact: The country containing {city} has a capital named",
        "Fact: The country containing {city} is named",
    ),
    (
        "capital_city",
        "Fact: The capital city of the country containing {city} is named",
        "Fact: The country containing {city} is named",
    ),
    (
        "national_capital",
        "Fact: The national capital of the country containing {city} is named",
        "Fact: The country containing {city} is named",
    ),
]


def checkpoint_payload(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(output_path)


def feature_key(feature: Feature) -> str:
    return f"L{feature[0]}F{feature[1]}"


def feature_dict(features: Iterable[Feature]) -> Dict[int, List[int]]:
    selected: Dict[int, List[int]] = {}
    for layer, feature in features:
        selected.setdefault(int(layer), []).append(int(feature))
    return {layer: sorted(set(values)) for layer, values in sorted(selected.items())}


def load_graph_feature_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        graph = json.load(handle)
    records = []
    for node in graph.get("nodes", []):
        match = re.fullmatch(r"layer_(\d+)_feature_(\d+)", str(node.get("id", "")))
        if not match:
            continue
        layer = int(match.group(1))
        if layer not in LAYERS:
            continue
        attribution = float(node.get("attribution", 0.0))
        if attribution <= 0:
            continue
        feature = int(match.group(2))
        records.append(
            {
                "key": feature_key((layer, feature)),
                "layer": layer,
                "feature": feature,
                "graph_attribution": attribution,
            }
        )
    records.sort(key=lambda row: (-abs(row["graph_attribution"]), row["layer"], row["feature"]))
    if not records:
        raise ValueError(f"No positive SAE feature nodes found in {path}")
    return records


def continuation_token(tokenizer, answer: str) -> Tuple[int, str]:
    """Resolve the first next-token representation, including its leading space."""
    token_ids = tokenizer.encode(" " + answer.strip(), add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Answer {answer!r} did not produce a token")
    token_id = int(token_ids[0])
    return token_id, tokenizer.decode([token_id])


def read_capitals(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"Location", "Type", "Answer", "DistractorAnswer", "sentence"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Expected columns {sorted(required)} in {path}")
    return rows


def candidate_cases(
    rows: Sequence[Dict[str, str]],
    seed: int,
    excluded_countries: Sequence[str],
    sae_corpus_prompts: set[str] | None = None,
) -> List[Dict[str, Any]]:
    by_country: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    excluded = {value.casefold() for value in excluded_countries}
    for row in rows:
        if row["Type"].strip().casefold() != "country":
            continue
        country = row["Location"].strip()
        if country.casefold() not in excluded:
            by_country[country].append(row)

    rng = np.random.default_rng(seed)
    countries = sorted(by_country)
    rng.shuffle(countries)
    cases = []
    for country in countries:
        country_rows = list(by_country[country])
        rng.shuffle(country_rows)
        for row in country_rows[:3]:
            city = row["DistractorAnswer"].strip()
            for template_name, capital_template, country_template in PROMPT_PAIRS:
                capital_prompt = capital_template.format(city=city)
                country_prompt = country_template.format(city=city)
                cases.append(
                    {
                        "country": country,
                        "capital": row["Answer"].strip(),
                        "city": city,
                        "template": template_name,
                        "capital_prompt": capital_prompt,
                        "country_prompt": country_prompt,
                        "capital_prompt_absent_from_sae_corpus": (
                            capital_prompt not in sae_corpus_prompts
                            if sae_corpus_prompts is not None
                            else capital_prompt != row["sentence"]
                        ),
                        "country_prompt_absent_from_sae_corpus": (
                            country_prompt not in sae_corpus_prompts
                            if sae_corpus_prompts is not None
                            else country_prompt != row["sentence"]
                        ),
                    }
                )
    return cases


def prepare_case(model, tokenizer, case: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(case)
    capital_id, capital_token = continuation_token(tokenizer, row["capital"])
    country_id, country_token = continuation_token(tokenizer, row["country"])
    if capital_id == country_id:
        row.update({"eligible": False, "ineligible_reason": "capital and country share first token"})
        return row

    capital_clean = baseline_condition(
        model, tokenizer, row["capital_prompt"], capital_id, country_id
    )
    country_clean = baseline_condition(
        model, tokenizer, row["country_prompt"], capital_id, country_id
    )
    capital_correct = bool(capital_clean["top_is_first"])
    country_correct = bool(country_clean["top_is_second"])
    reasons = []
    if not capital_correct:
        reasons.append("capital prompt did not predict capital prefix")
    if not country_correct:
        reasons.append("inverse prompt did not predict country prefix")
    row.update(
        {
            "capital_id": capital_id,
            "country_id": country_id,
            "capital_token": capital_token,
            "country_token": country_token,
            "capital_clean": capital_clean,
            "country_clean": country_clean,
            "eligible": capital_correct and country_correct,
            "ineligible_reason": "; ".join(reasons) if reasons else None,
        }
    )
    return row


def select_eligible_cases(
    model,
    tokenizer,
    cases: Sequence[Dict[str, Any]],
    required: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected: Dict[str, Dict[str, Any]] = {}
    screened = []
    for index, case in enumerate(cases, start=1):
        if case["country"] in selected:
            continue
        print(
            f"[baseline {index:03d}] {case['city']} -> {case['capital']} / "
            f"{case['country']} ({case['template']})"
        )
        prepared = prepare_case(model, tokenizer, case)
        screened.append(prepared)
        if prepared.get("eligible"):
            selected[prepared["country"]] = prepared
            if len(selected) == required:
                break
    return list(selected.values()), screened


def activation_mean(result: Dict[str, Any], feature: Feature) -> float:
    layer, index = feature
    layer_values = result.get("feature_activations", {}).get(layer, {})
    return float(layer_values.get(index, {}).get("mean", 0.0))


def evaluate_panel(
    model,
    tokenizer,
    saes,
    panel: Sequence[Feature],
    cases: Sequence[Dict[str, Any]],
    verbose: bool,
) -> List[Dict[str, Any]]:
    selected = feature_dict(panel)
    layers = sorted(selected)
    rows = []
    for case in cases:
        targets = [case["capital_token"], case["country_token"]]
        with suppress_output(not verbose):
            capital_result = run_inhibition_intervention(
                model,
                tokenizer,
                case["capital_prompt"],
                layers,
                saes,
                selected,
                targets,
                position_spec="last",
            )
            country_result = run_inhibition_intervention(
                model,
                tokenizer,
                case["country_prompt"],
                layers,
                saes,
                selected,
                targets,
                position_spec="last",
            )
        capital = condition_from_result(
            capital_result,
            case["capital_token"],
            case["country_token"],
            case["capital_id"],
            case["country_id"],
        )
        country = condition_from_result(
            country_result,
            case["capital_token"],
            case["country_token"],
            case["capital_id"],
            case["country_id"],
        )
        capital_delta = float(capital["gap"] - case["capital_clean"]["gap"])
        country_delta = float(country["gap"] - case["country_clean"]["gap"])
        record = {
            "country": case["country"],
            "capital": case["capital"],
            "city": case["city"],
            "capital_prompt_delta": capital_delta,
            "inverse_country_prompt_delta": country_delta,
            "relation_specific_difference": capital_delta - country_delta,
            "capital_prompt_flipped_to_country": bool(capital["top_is_second"]),
            "capital_prompt_retained_capital": bool(capital["top_is_first"]),
            "inverse_prompt_retained_country": bool(country["top_is_second"]),
        }
        if len(panel) == 1:
            feature = panel[0]
            record["capital_activation"] = activation_mean(capital_result, feature)
            record["inverse_country_activation"] = activation_mean(country_result, feature)
            record["activation_difference"] = (
                record["capital_activation"] - record["inverse_country_activation"]
            )
        rows.append(record)
    return rows


def summarise_rows(rows: Sequence[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    capital = np.asarray([row["capital_prompt_delta"] for row in rows], dtype=float)
    inverse = np.asarray([row["inverse_country_prompt_delta"] for row in rows], dtype=float)
    paired = capital - inverse
    summary = {
        "eligible_cases": len(rows),
        "mean_capital_prompt_delta": float(capital.mean()),
        "bootstrap_95_ci_mean_capital_prompt_delta": list(bootstrap_mean_ci(capital, seed)),
        "mean_inverse_country_prompt_delta": float(inverse.mean()),
        "bootstrap_95_ci_mean_inverse_country_prompt_delta": list(
            bootstrap_mean_ci(inverse, seed + 1)
        ),
        "mean_relation_specific_difference": float(paired.mean()),
        "bootstrap_95_ci_mean_relation_specific_difference": list(
            bootstrap_mean_ci(paired, seed + 2)
        ),
        "fraction_capital_delta_negative": float(np.mean(capital < 0)),
        "fraction_relation_specific_difference_negative": float(np.mean(paired < 0)),
        "capital_to_country_flip_fraction": float(
            np.mean([row["capital_prompt_flipped_to_country"] for row in rows])
        ),
        "capital_retention_fraction": float(
            np.mean([row["capital_prompt_retained_capital"] for row in rows])
        ),
        "inverse_country_retention_fraction": float(
            np.mean([row["inverse_prompt_retained_country"] for row in rows])
        ),
    }
    if "activation_difference" in rows[0]:
        activation = np.asarray([row["activation_difference"] for row in rows], dtype=float)
        summary.update(
            {
                "mean_capital_minus_inverse_activation": float(activation.mean()),
                "fraction_capital_activation_greater": float(np.mean(activation > 0)),
            }
        )
    return summary


def rank_feature_results(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    supportive = [row for row in results if row["summary"]["mean_capital_prompt_delta"] < 0]
    remaining = [row for row in results if row not in supportive]

    def key(row: Dict[str, Any]) -> Tuple[float, float, float, int, int]:
        summary = row["summary"]
        return (
            float(summary["mean_relation_specific_difference"]),
            float(summary["mean_capital_prompt_delta"]),
            -float(row["graph_attribution"]),
            int(row["layer"]),
            int(row["feature"]),
        )

    return sorted(supportive, key=key) + sorted(remaining, key=key)


def matched_random_panel(
    candidates: Sequence[Feature], reference: Sequence[Feature], rng: np.random.Generator
) -> List[Feature]:
    by_layer: Dict[int, List[int]] = defaultdict(list)
    for layer, feature in candidates:
        by_layer[layer].append(feature)
    requested = Counter(layer for layer, _ in reference)
    panel = []
    for layer, count in sorted(requested.items()):
        chosen = rng.choice(np.asarray(sorted(by_layer[layer])), size=count, replace=False)
        panel.extend((layer, int(value)) for value in chosen)
    return sorted(panel)


def build_panels(
    ranked: Sequence[Feature],
    candidates: Sequence[Feature],
    panel_sizes: Sequence[int],
    primary_size: int,
    random_panels: int,
    seed: int,
) -> List[Dict[str, Any]]:
    panels = []
    for size in sorted(set(panel_sizes)):
        panels.append(
            {
                "name": f"top_{size}" + ("_primary" if size == primary_size else ""),
                "kind": "discovery_ranked_prefix",
                "features": list(ranked[: min(size, len(ranked))]),
            }
        )
    primary = list(ranked[: min(primary_size, len(ranked))])
    panels.extend(
        [
            {"name": "all_positive_graph", "kind": "full_graph", "features": list(candidates)},
            {
                "name": f"bottom_{primary_size}",
                "kind": "reverse_rank_control",
                "features": list(ranked[-len(primary) :]),
            },
        ]
    )
    rng = np.random.default_rng(seed)
    for index in range(random_panels):
        panels.append(
            {
                "name": f"random_matched_{index + 1:02d}",
                "kind": "layer_count_matched_random_control",
                "features": matched_random_panel(candidates, primary, rng),
            }
        )
    return panels


def serialise_panel(panel: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": panel["name"],
        "kind": panel["kind"],
        "feature_count": len(panel["features"]),
        "features": [
            {"layer": layer, "feature": feature, "key": feature_key((layer, feature))}
            for layer, feature in panel["features"]
        ],
        "layer_counts": dict(Counter(layer for layer, _ in panel["features"])),
    }


def protocol_signature(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model_config": args.model_config,
        "sae_config": args.sae_config,
        "capitals_csv": args.capitals_csv,
        "sae_corpus_csv": args.sae_corpus_csv,
        "graph": args.graph,
        "seed": args.seed,
        "discovery_cases": args.discovery_cases,
        "confirmation_cases": args.confirmation_cases,
        "panel_sizes": list(args.panel_sizes),
        "primary_panel_size": args.primary_panel_size,
        "random_panels": args.random_panels,
        "excluded_countries": sorted(args.exclude_country),
        "positions": "last",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capital-relation SAE feature screen")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--sae-config", required=True)
    parser.add_argument("--capitals-csv", default="data/capitals_data.csv")
    parser.add_argument(
        "--sae-corpus-csv",
        default="data/capitals_data.csv",
        help="Prompt CSV used to train the evaluated SAE, for exact-overlap auditing.",
    )
    parser.add_argument("--graph", required=True)
    parser.add_argument("--exclude-country", action="append", default=[])
    parser.add_argument("--discovery-cases", type=int, default=8)
    parser.add_argument("--confirmation-cases", type=int, default=16)
    parser.add_argument("--panel-sizes", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    parser.add_argument("--primary-panel-size", type=int, default=3)
    parser.add_argument("--random-panels", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4787)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/capitals_relation_pilot/capitals_relation_feature_screen.json",
    )
    args = parser.parse_args()
    if not args.exclude_country:
        args.exclude_country = ["Jordan"]

    if args.primary_panel_size not in args.panel_sizes:
        raise ValueError("--primary-panel-size must also appear in --panel-sizes")
    required = args.discovery_cases + args.confirmation_cases
    repo_root = get_repo_root()
    graph_path = resolve_path(args.graph, repo_root)
    config_path = resolve_path(args.sae_config, repo_root)
    capitals_path = resolve_path(args.capitals_csv, repo_root)
    sae_corpus_path = resolve_path(args.sae_corpus_csv, repo_root)
    output_path = resolve_path(args.output, repo_root)
    signature = protocol_signature(args)
    started = time.time()

    payload = None
    if output_path.exists() and not args.overwrite:
        with output_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("protocol_signature") != signature:
            raise ValueError(
                f"Existing output {output_path} uses a different protocol; choose a new path "
                "or pass --overwrite."
            )
        payload = existing
        print(f"Resuming partial output: {output_path}")
    if payload is None:
        payload = {
            "status": "initialising",
            "protocol_signature": signature,
            "method": {
                "candidate_definition": (
                    "positive-attribution SAE features from one Amman-over-Jordan capital graph"
                ),
                "primary_gap": "logit(capital prefix) minus logit(country prefix)",
                "screening_estimand": (
                    "capital-prompt inhibition delta minus inverse-country-prompt inhibition delta"
                ),
                "desired_direction": "negative",
                "selection_data": "discovery countries only",
                "confirmation_data_used_for_ranking": False,
                "primary_confirmation_panel": f"top_{args.primary_panel_size}_primary",
                "primary_success_rule": (
                    "capital-prompt mean delta and relation-specific mean are negative, with both "
                    "bootstrap 95% confidence intervals wholly below zero"
                ),
                "scope": "final-token, error-preserving feature inhibition",
                "categorical_country_flip_required": False,
            },
            "discovery": {"feature_results": []},
            "confirmation": {"panels": []},
        }
        checkpoint_payload(payload, output_path)

    graph_records = load_graph_feature_records(graph_path)
    candidates = [(row["layer"], row["feature"]) for row in graph_records]
    payload["candidate_features"] = graph_records
    payload["candidate_feature_count"] = len(candidates)
    payload["candidate_layer_counts"] = dict(Counter(layer for layer, _ in candidates))
    checkpoint_payload(payload, output_path)

    print("Loading model and capitals SAEs...")
    model, tokenizer, _ = load_model_and_tokenizer(resolve_path(args.model_config, repo_root))
    saes = load_domain_saes(model, config_path)

    if "case_selection" not in payload:
        source_rows = read_capitals(capitals_path)
        with sae_corpus_path.open("r", encoding="utf-8", newline="") as handle:
            sae_corpus_prompts = {row["sentence"] for row in csv.DictReader(handle)}
        cases = candidate_cases(
            source_rows,
            args.seed,
            args.exclude_country,
            sae_corpus_prompts=sae_corpus_prompts,
        )
        eligible, screened = select_eligible_cases(model, tokenizer, cases, required)
        if len(eligible) < required:
            raise ValueError(
                f"Only {len(eligible)} distinct countries produced correct capital and inverse "
                f"baselines; {required} are required. Reduce the split sizes if necessary."
            )
        discovery = eligible[: args.discovery_cases]
        confirmation = eligible[args.discovery_cases : required]
        payload["case_selection"] = {
            "screened_case_count": len(screened),
            "eligible_country_count": len(eligible),
            "screened_cases": screened,
            "discovery_cases": discovery,
            "confirmation_cases": confirmation,
            "countries_disjoint_between_splits": True,
            "graph_entity_excluded": list(args.exclude_country),
            "split_frozen_before_feature_interventions": True,
            "all_selected_prompts_absent_from_exact_sae_corpus": bool(
                all(
                    row["capital_prompt_absent_from_sae_corpus"]
                    and row["country_prompt_absent_from_sae_corpus"]
                    for row in eligible
                )
            ),
            "sae_corpus_csv": str(sae_corpus_path),
            "sae_corpus_prompt_count": len(sae_corpus_prompts),
        }
        if not payload["case_selection"]["all_selected_prompts_absent_from_exact_sae_corpus"]:
            raise ValueError("A selected capitals relation prompt overlaps the SAE training corpus")
        checkpoint_payload(payload, output_path)
    discovery = payload["case_selection"]["discovery_cases"]
    confirmation = payload["case_selection"]["confirmation_cases"]

    payload["status"] = "discovery"
    completed = {row["key"] for row in payload["discovery"]["feature_results"]}
    for index, record in enumerate(graph_records, start=1):
        if record["key"] in completed:
            print(f"[feature {index:02d}/{len(graph_records)}] {record['key']} already complete")
            continue
        print(f"[feature {index:02d}/{len(graph_records)}] screening {record['key']}")
        feature = (record["layer"], record["feature"])
        rows = evaluate_panel(model, tokenizer, saes, [feature], discovery, args.verbose)
        result = dict(record)
        result["summary"] = summarise_rows(rows, args.seed + index * 10)
        result["case_effects"] = rows
        payload["discovery"]["feature_results"].append(result)
        checkpoint_payload(payload, output_path)

    ranked_results = rank_feature_results(payload["discovery"]["feature_results"])
    for rank, row in enumerate(ranked_results, start=1):
        row["discovery_rank"] = rank
    ranked = [(row["layer"], row["feature"]) for row in ranked_results]
    payload["discovery"]["feature_results"] = ranked_results
    payload["discovery"]["frozen_feature_order"] = [feature_key(value) for value in ranked]
    payload["discovery"]["ranking_rule"] = (
        "Features with negative capital-prompt effects are ordered first, then by ascending "
        "relation-specific difference, capital effect, graph attribution and stable feature ID."
    )
    payload["discovery"]["confirmation_was_run_when_order_frozen"] = False
    checkpoint_payload(payload, output_path)

    panels = build_panels(
        ranked,
        candidates,
        args.panel_sizes,
        args.primary_panel_size,
        args.random_panels,
        args.seed + 5000,
    )
    payload["status"] = "confirmation"
    completed_panels = {row["name"] for row in payload["confirmation"]["panels"]}
    for index, panel in enumerate(panels, start=1):
        if panel["name"] in completed_panels:
            print(f"[panel {index:02d}/{len(panels)}] {panel['name']} already complete")
            continue
        print(
            f"[panel {index:02d}/{len(panels)}] confirming {panel['name']} "
            f"({len(panel['features'])} features)"
        )
        rows = evaluate_panel(model, tokenizer, saes, panel["features"], confirmation, args.verbose)
        result = serialise_panel(panel)
        result["summary"] = summarise_rows(rows, args.seed + 10000 + index * 10)
        result["case_effects"] = rows
        payload["confirmation"]["panels"].append(result)
        checkpoint_payload(payload, output_path)

    primary_name = f"top_{args.primary_panel_size}_primary"
    primary = next(row for row in payload["confirmation"]["panels"] if row["name"] == primary_name)
    summary = primary["summary"]
    capital_ci = summary["bootstrap_95_ci_mean_capital_prompt_delta"]
    paired_ci = summary["bootstrap_95_ci_mean_relation_specific_difference"]
    success = bool(
        summary["mean_capital_prompt_delta"] < 0
        and capital_ci[1] < 0
        and summary["mean_relation_specific_difference"] < 0
        and paired_ci[1] < 0
    )
    random_effects = [
        row["summary"]["mean_relation_specific_difference"]
        for row in payload["confirmation"]["panels"]
        if row["kind"] == "layer_count_matched_random_control"
    ]
    payload["confirmation"]["primary_result"] = {
        "panel": primary_name,
        "supports_capital_relation_selectivity_under_predeclared_rule": success,
        "summary": summary,
        "random_control_mean_relation_specific_differences": random_effects,
        "primary_more_negative_than_random_control_fraction": (
            float(np.mean(summary["mean_relation_specific_difference"] < np.asarray(random_effects)))
            if random_effects
            else None
        ),
    }
    payload["status"] = "complete"
    payload["runtime_seconds"] = time.time() - started
    checkpoint_payload(payload, output_path)

    print("\nPrimary confirmation result")
    print(f"  panel: {primary_name}")
    print(f"  capital-prompt mean delta: {summary['mean_capital_prompt_delta']:+.4f}")
    print(f"  inverse-country mean delta: {summary['mean_inverse_country_prompt_delta']:+.4f}")
    print(
        "  relation-specific mean: "
        f"{summary['mean_relation_specific_difference']:+.4f} "
        f"(95% CI [{paired_ci[0]:+.4f}, {paired_ci[1]:+.4f}])"
    )
    print(f"  capital-to-country top-token flips: {summary['capital_to_country_flip_fraction']:.1%}")
    print(f"  predeclared success rule met: {success}")
    print(f"Saved screen to {output_path}")


if __name__ == "__main__":
    main()

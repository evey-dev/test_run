"""Discover and confirm carry-selective SAE feature panels.

The screen uses only final-token interventions because the mathematics SAEs and
the source attribution graph were both fitted/evaluated at that position. A
discovery split ranks graph features by the paired causal contrast

    carry-target gap delta - matched no-carry gap delta,

where a negative value is carry-selective. The feature ordering is frozen before
any confirmation intervention is run. A fresh confirmation split then evaluates
the predeclared top-10 panel, a cumulative dose response, layer groups, and
deterministic matched-size controls.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.data_utils import get_repo_root, resolve_path
from src.heldout_validation import (
    LAYERS,
    baseline_condition,
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
    """Atomically replace a checkpoint so an interrupted Drive write is resumable."""
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
    return {layer: sorted(set(indices)) for layer, indices in sorted(selected.items())}


def load_graph_feature_records(
    path: Path,
    layers: Sequence[int],
    sign: str = "positive",
) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        graph = json.load(handle)

    records = []
    for node in graph.get("nodes", []):
        match = re.fullmatch(r"layer_(\d+)_feature_(\d+)", str(node.get("id", "")))
        if not match:
            continue
        layer = int(match.group(1))
        if layer not in layers:
            continue
        attribution = float(node.get("attribution", 0.0))
        if sign == "positive" and attribution <= 0:
            continue
        if sign == "negative" and attribution >= 0:
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
        raise ValueError(f"No {sign} graph features found in {path}")
    return records


def case_key(case: Dict[str, Any]) -> str:
    return f"{case['source_a']}+{case['b']}->{case['target_a']}+{case['b']}"


def select_fresh_case_pool(
    addition_csv: Path,
    seed: int,
    required: int,
    excluded_seed: int,
    excluded_count: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return deterministic cases excluding the previously inspected benchmark."""
    excluded = generated_math_cases(excluded_count, addition_csv, excluded_seed)
    excluded_keys = {case_key(case) for case in excluded}
    # The current deterministic addition corpus has 149 eligible graph-held-out pairs.
    # A pool of 120 leaves ample room after excluding the original benchmark.
    candidates = generated_math_cases(120, addition_csv, seed)
    fresh = [case for case in candidates if case_key(case) not in excluded_keys]
    if len(fresh) < required:
        raise ValueError(
            f"Only {len(fresh)} fresh cases remain after exclusions; {required} are required"
        )
    return fresh, sorted(excluded_keys)


def prepare_case(model, tokenizer, case: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(case)
    row["case_key"] = case_key(case)
    correct_id = first_token_id(tokenizer, case["correct_digit"])
    dropped_id = first_token_id(tokenizer, case["dropped_carry_digit"])
    correct_token = tokenizer.decode([correct_id])
    dropped_token = tokenizer.decode([dropped_id])
    source_ids = tokenizer(case["source_prompt"], return_tensors="pt")["input_ids"]
    target_ids = tokenizer(case["target_prompt"], return_tensors="pt")["input_ids"]

    row.update(
        {
            "correct_id": correct_id,
            "dropped_id": dropped_id,
            "correct_token": correct_token,
            "dropped_token": dropped_token,
            "source_token_count": int(source_ids.shape[1]),
            "target_token_count": int(target_ids.shape[1]),
        }
    )
    if source_ids.shape[1] != target_ids.shape[1]:
        row["eligible"] = False
        row["ineligible_reason"] = "source and target token lengths differ"
        return row

    clean = baseline_condition(model, tokenizer, case["target_prompt"], correct_id, dropped_id)
    source_clean = baseline_condition(model, tokenizer, case["source_prompt"], dropped_id, correct_id)
    row["clean"] = clean
    row["source_clean"] = source_clean
    row["eligible"] = bool(clean["top_is_first"] and source_clean["top_is_first"])
    if not row["eligible"]:
        row["ineligible_reason"] = "clean target or source top prediction was not expected"
    return row


def public_case_record(case: Dict[str, Any]) -> Dict[str, Any]:
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
        "source_token_count",
        "target_token_count",
        "eligible",
        "ineligible_reason",
        "clean",
        "source_clean",
    ]
    return {key: case[key] for key in keep if key in case}


def singleton_activation(result: Dict[str, Any], feature: Feature) -> float:
    layer, feature_index = feature
    layer_values = result.get("feature_activations", {}).get(layer, {})
    feature_values = layer_values.get(feature_index, {})
    return float(feature_values.get("max", 0.0))


def evaluate_panel(
    model,
    tokenizer,
    saes,
    panel: Sequence[Feature],
    cases: Sequence[Dict[str, Any]],
    position_spec: str,
    verbose: bool,
) -> List[Dict[str, Any]]:
    features = feature_dict(panel)
    layers = sorted(features)
    singleton = panel[0] if len(panel) == 1 else None
    rows = []

    for case in cases:
        if not case.get("eligible"):
            continue
        with suppress_output(not verbose):
            target_result = run_inhibition_intervention(
                model,
                tokenizer,
                case["target_prompt"],
                layers,
                saes,
                features,
                [case["correct_token"], case["dropped_token"]],
                position_spec=position_spec,
            )
            control_result = run_inhibition_intervention(
                model,
                tokenizer,
                case["source_prompt"],
                layers,
                saes,
                features,
                [case["dropped_token"], case["correct_token"]],
                position_spec=position_spec,
            )

        target = condition_from_result(
            target_result,
            case["correct_token"],
            case["dropped_token"],
            case["correct_id"],
            case["dropped_id"],
        )
        control = condition_from_result(
            control_result,
            case["dropped_token"],
            case["correct_token"],
            case["dropped_id"],
            case["correct_id"],
        )
        target_delta = float(target["gap"] - case["clean"]["gap"])
        control_delta = float(control["gap"] - case["source_clean"]["gap"])
        row: Dict[str, Any] = {
            "case_key": case["case_key"],
            "target_delta": target_delta,
            "control_delta": control_delta,
            "paired_difference": target_delta - control_delta,
            "target_top_transferred": bool(target["top_is_second"]),
            "control_top_transferred": bool(control["top_is_second"]),
        }
        if singleton is not None:
            row["target_activation"] = singleton_activation(target_result, singleton)
            row["control_activation"] = singleton_activation(control_result, singleton)
        rows.append(row)
    return rows


def summarise_panel(rows: Sequence[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    if not rows:
        return {"eligible_cases": 0}
    target = np.asarray([row["target_delta"] for row in rows], dtype=float)
    control = np.asarray([row["control_delta"] for row in rows], dtype=float)
    paired = target - control
    summary: Dict[str, Any] = {
        "eligible_cases": len(rows),
        "mean_target_delta": float(target.mean()),
        "bootstrap_95_ci_mean_target_delta": list(bootstrap_mean_ci(target, seed)),
        "mean_no_carry_control_delta": float(control.mean()),
        "bootstrap_95_ci_mean_no_carry_control_delta": list(
            bootstrap_mean_ci(control, seed + 1)
        ),
        "mean_paired_difference": float(paired.mean()),
        "bootstrap_95_ci_mean_paired_difference": list(
            bootstrap_mean_ci(paired, seed + 2)
        ),
        "fraction_target_more_negative_than_control": float(np.mean(target < control)),
        "fraction_target_delta_negative": float(np.mean(target < 0)),
        "target_top_prediction_transfer_fraction": float(
            np.mean([row["target_top_transferred"] for row in rows])
        ),
        "control_top_prediction_transfer_fraction": float(
            np.mean([row["control_top_transferred"] for row in rows])
        ),
    }
    if "target_activation" in rows[0]:
        target_activation = np.asarray([row["target_activation"] for row in rows], dtype=float)
        control_activation = np.asarray([row["control_activation"] for row in rows], dtype=float)
        summary.update(
            {
                "mean_target_activation": float(target_activation.mean()),
                "mean_control_activation": float(control_activation.mean()),
                "target_active_fraction": float(np.mean(target_activation > 1e-6)),
                "control_active_fraction": float(np.mean(control_activation > 1e-6)),
            }
        )
    return summary


def rank_feature_results(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Freeze a carry-supporting ordering using discovery outcomes only."""
    active_support = [
        row
        for row in results
        if row["summary"].get("mean_target_delta", 0.0) < 0
        and row["summary"].get("target_active_fraction", 0.0) >= 0.25
    ]
    remaining = [row for row in results if row not in active_support]

    def key(row: Dict[str, Any]) -> Tuple[float, float, float, int, int]:
        summary = row["summary"]
        return (
            float(summary.get("mean_paired_difference", 0.0)),
            float(summary.get("mean_target_delta", 0.0)),
            -float(row.get("graph_attribution", 0.0)),
            int(row["layer"]),
            int(row["feature"]),
        )

    return sorted(active_support, key=key) + sorted(remaining, key=key)


def matched_random_panel(
    candidate_features: Sequence[Feature],
    reference_panel: Sequence[Feature],
    rng: np.random.Generator,
) -> List[Feature]:
    by_layer: Dict[int, List[int]] = {}
    for layer, feature in candidate_features:
        by_layer.setdefault(layer, []).append(feature)
    requested = Counter(layer for layer, _ in reference_panel)
    panel: List[Feature] = []
    for layer, count in sorted(requested.items()):
        pool = np.asarray(sorted(by_layer[layer]), dtype=int)
        chosen = rng.choice(pool, size=count, replace=False)
        panel.extend((layer, int(feature)) for feature in chosen)
    return sorted(panel)


def build_confirmation_panels(
    ranked_features: Sequence[Feature],
    candidate_features: Sequence[Feature],
    panel_sizes: Sequence[int],
    primary_panel_size: int,
    random_panels: int,
    seed: int,
) -> List[Dict[str, Any]]:
    panels = []
    for size in sorted(set(int(value) for value in panel_sizes)):
        actual = min(size, len(ranked_features))
        name = f"top_{size}"
        if size == primary_panel_size:
            name += "_primary"
        panels.append(
            {
                "name": name,
                "kind": "discovery_ranked_prefix",
                "requested_size": size,
                "features": list(ranked_features[:actual]),
            }
        )

    primary = list(ranked_features[: min(primary_panel_size, len(ranked_features))])
    panels.extend(
        [
            {
                "name": "all_positive_graph",
                "kind": "full_graph_comparator",
                "requested_size": len(candidate_features),
                "features": list(candidate_features),
            },
            {
                "name": f"bottom_{primary_panel_size}",
                "kind": "reverse_rank_control",
                "requested_size": primary_panel_size,
                "features": list(ranked_features[-len(primary) :]),
            },
        ]
    )
    for layer in sorted({layer for layer, _ in candidate_features}):
        layer_features = [feature for feature in candidate_features if feature[0] == layer]
        panels.append(
            {
                "name": f"layer_{layer}_positive_graph",
                "kind": "layer_group",
                "requested_size": len(layer_features),
                "features": layer_features,
            }
        )

    rng = np.random.default_rng(seed)
    for index in range(random_panels):
        panels.append(
            {
                "name": f"random_matched_{index + 1:02d}",
                "kind": "layer_count_matched_random_control",
                "requested_size": len(primary),
                "features": matched_random_panel(candidate_features, primary, rng),
            }
        )
    return panels


def serialise_panel(panel: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(panel)
    result["features"] = [
        {"layer": layer, "feature": feature, "key": feature_key((layer, feature))}
        for layer, feature in panel["features"]
    ]
    result["feature_count"] = len(panel["features"])
    result["layer_counts"] = dict(Counter(layer for layer, _ in panel["features"]))
    return result


def protocol_signature(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "protocol_version": 1,
        "model_config": str(args.model_config),
        "sae_config": str(args.sae_config),
        "graph": str(args.graph),
        "positions": args.positions,
        "discovery_seed": args.seed,
        "excluded_previous_seed": args.exclude_seed,
        "excluded_previous_cases": args.exclude_cases,
        "discovery_cases": args.discovery_cases,
        "confirmation_cases": args.confirmation_cases,
        "panel_sizes": list(args.panel_sizes),
        "primary_panel_size": args.primary_panel_size,
        "random_panels": args.random_panels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Carry-selective SAE feature discovery and confirmation")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--sae-config", default="configs/sae_math_topk256_config.yaml")
    parser.add_argument(
        "--graph",
        default="outputs/topk_math_retrain/math_topk256_carry_58_83_4v3_graph.json",
    )
    parser.add_argument("--positions", default="last")
    parser.add_argument("--discovery-cases", type=int, default=8)
    parser.add_argument("--confirmation-cases", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1787)
    parser.add_argument("--exclude-seed", type=int, default=787)
    parser.add_argument("--exclude-cases", type=int, default=12)
    parser.add_argument("--panel-sizes", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    parser.add_argument("--primary-panel-size", type=int, default=10)
    parser.add_argument("--random-panels", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/topk_math_followup/math_topk256_carry_feature_screen.json",
    )
    args = parser.parse_args()

    if args.positions.lower() not in {"last", "final"}:
        raise ValueError(
            "This confirmatory screen must use --positions last because the SAE corpus and "
            "attribution graph are final-position analyses"
        )
    if args.primary_panel_size not in args.panel_sizes:
        raise ValueError("--primary-panel-size must also appear in --panel-sizes")

    repo_root = get_repo_root()
    graph_path = resolve_path(args.graph, repo_root)
    config_path = resolve_path(args.sae_config, repo_root)
    output_path = resolve_path(args.output, repo_root)
    signature = protocol_signature(args)
    started = time.time()

    if output_path.exists() and not args.overwrite:
        with output_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("protocol_signature") != signature:
            raise ValueError(
                f"Existing partial output {output_path} uses a different protocol. "
                "Use --overwrite or choose another --output path."
            )
        print(f"Resuming partial output: {output_path}")
    else:
        payload = {
            "status": "initialising",
            "protocol_signature": signature,
            "method": {
                "candidate_definition": "positive-attribution features in one 58+83 carry graph",
                "screening_estimand": "carry target gap delta minus matched no-carry control gap delta",
                "desired_screening_direction": "negative",
                "selection_data": "discovery split only",
                "confirmation_data_used_for_ranking": False,
                "primary_confirmation_panel": f"top_{args.primary_panel_size}_primary",
                "primary_success_rule": (
                    "negative mean paired difference, negative upper endpoint of its bootstrap 95% CI, "
                    "and negative mean carry-target delta"
                ),
                "scope": (
                    "Final-token error-preserving inhibition only; this matches the position used for "
                    "SAE training and graph construction."
                ),
            },
            "discovery": {"feature_results": []},
            "confirmation": {"panels": []},
        }
        checkpoint_payload(payload, output_path)

    graph_records = load_graph_feature_records(graph_path, LAYERS, sign="positive")
    candidate_features = [(row["layer"], row["feature"]) for row in graph_records]
    payload["candidate_features"] = graph_records
    payload["candidate_feature_count"] = len(graph_records)
    payload["candidate_layer_counts"] = dict(Counter(layer for layer, _ in candidate_features))

    required = args.discovery_cases + args.confirmation_cases
    case_pool, excluded_keys = select_fresh_case_pool(
        repo_root / "data/addition_data.csv",
        args.seed,
        required=required,
        excluded_seed=args.exclude_seed,
        excluded_count=args.exclude_cases,
    )

    print("Loading model and selected TopK SAEs...")
    model, tokenizer, _ = load_model_and_tokenizer(repo_root / args.model_config)
    saes = load_domain_saes(model, config_path)

    prepared = []
    for index, case in enumerate(case_pool, start=1):
        print(f"[baseline {index:02d}/{len(case_pool)}] {case_key(case)}")
        prepared.append(prepare_case(model, tokenizer, case))
        if sum(bool(row.get("eligible")) for row in prepared) >= required:
            break
    eligible = [row for row in prepared if row.get("eligible")]
    if len(eligible) < required:
        raise ValueError(f"Only {len(eligible)} of {required} required fresh cases passed baselines")
    discovery_cases = eligible[: args.discovery_cases]
    confirmation_cases = eligible[args.discovery_cases : required]
    payload["case_selection"] = {
        "excluded_previously_inspected_case_keys": excluded_keys,
        "baseline_screened_cases": [public_case_record(row) for row in prepared],
        "discovery_case_keys": [row["case_key"] for row in discovery_cases],
        "confirmation_case_keys": [row["case_key"] for row in confirmation_cases],
        "split_frozen_before_feature_interventions": True,
    }
    payload["status"] = "discovery"
    checkpoint_payload(payload, output_path)

    completed = {row["key"] for row in payload["discovery"]["feature_results"]}
    for index, feature_record in enumerate(graph_records, start=1):
        key = feature_record["key"]
        if key in completed:
            print(f"[feature {index:02d}/{len(graph_records)}] {key} already complete")
            continue
        feature = (feature_record["layer"], feature_record["feature"])
        print(f"[feature {index:02d}/{len(graph_records)}] screening {key}")
        rows = evaluate_panel(
            model,
            tokenizer,
            saes,
            [feature],
            discovery_cases,
            args.positions,
            args.verbose,
        )
        result = dict(feature_record)
        result["summary"] = summarise_panel(rows, args.seed + index * 10)
        result["case_effects"] = rows
        payload["discovery"]["feature_results"].append(result)
        checkpoint_payload(payload, output_path)

    ranked_results = rank_feature_results(payload["discovery"]["feature_results"])
    for rank, row in enumerate(ranked_results, start=1):
        row["discovery_rank"] = rank
        row["passes_descriptive_stability_filter"] = bool(
            row["summary"].get("mean_target_delta", 0.0) < 0
            and row["summary"].get("mean_paired_difference", 0.0) < 0
            and row["summary"].get("target_active_fraction", 0.0) >= 0.25
            and row["summary"].get("fraction_target_more_negative_than_control", 0.0) >= 0.625
        )
    ranked_features = [(row["layer"], row["feature"]) for row in ranked_results]
    payload["discovery"]["feature_results"] = ranked_results
    payload["discovery"]["frozen_feature_order"] = [feature_key(value) for value in ranked_features]
    payload["discovery"]["ranking_rule"] = (
        "Features active on at least 25% of discovery carry targets and with a negative mean "
        "carry-target effect are ordered first; within groups, order is ascending discovery "
        "paired difference, then target delta, then graph attribution and stable feature ID."
    )
    payload["discovery"]["confirmation_was_run_when_order_frozen"] = False
    checkpoint_payload(payload, output_path)

    panels = build_confirmation_panels(
        ranked_features,
        candidate_features,
        args.panel_sizes,
        args.primary_panel_size,
        args.random_panels,
        args.seed + 5000,
    )
    existing_panels = {row["name"] for row in payload["confirmation"]["panels"]}
    payload["status"] = "confirmation"
    for index, panel in enumerate(panels, start=1):
        if panel["name"] in existing_panels:
            print(f"[panel {index:02d}/{len(panels)}] {panel['name']} already complete")
            continue
        print(
            f"[panel {index:02d}/{len(panels)}] confirming {panel['name']} "
            f"({len(panel['features'])} features)"
        )
        rows = evaluate_panel(
            model,
            tokenizer,
            saes,
            panel["features"],
            confirmation_cases,
            args.positions,
            args.verbose,
        )
        result = serialise_panel(panel)
        result["summary"] = summarise_panel(rows, args.seed + 10000 + index * 10)
        result["case_effects"] = rows
        payload["confirmation"]["panels"].append(result)
        checkpoint_payload(payload, output_path)

    primary_name = f"top_{args.primary_panel_size}_primary"
    primary = next(row for row in payload["confirmation"]["panels"] if row["name"] == primary_name)
    primary_summary = primary["summary"]
    paired_ci = primary_summary["bootstrap_95_ci_mean_paired_difference"]
    primary_success = bool(
        primary_summary["mean_paired_difference"] < 0
        and paired_ci[1] < 0
        and primary_summary["mean_target_delta"] < 0
    )
    random_effects = [
        row["summary"]["mean_paired_difference"]
        for row in payload["confirmation"]["panels"]
        if row["kind"] == "layer_count_matched_random_control"
    ]
    payload["confirmation"]["primary_result"] = {
        "panel": primary_name,
        "supports_carry_selectivity_under_predeclared_rule": primary_success,
        "summary": primary_summary,
        "random_control_mean_paired_differences": random_effects,
        "primary_more_negative_than_random_control_fraction": (
            float(np.mean(primary_summary["mean_paired_difference"] < np.asarray(random_effects)))
            if random_effects
            else None
        ),
    }
    payload["status"] = "complete"
    payload["runtime_seconds"] = time.time() - started
    checkpoint_payload(payload, output_path)

    print("\nPrimary confirmation result")
    print(f"  panel: {primary_name}")
    print(f"  carry-target mean delta: {primary_summary['mean_target_delta']:+.4f}")
    print(f"  no-carry mean delta: {primary_summary['mean_no_carry_control_delta']:+.4f}")
    print(
        "  paired carry-minus-control: "
        f"{primary_summary['mean_paired_difference']:+.4f} "
        f"(95% CI [{paired_ci[0]:+.4f}, {paired_ci[1]:+.4f}])"
    )
    print(f"  predeclared success rule met: {primary_success}")
    print(f"Saved screen to {output_path}")


if __name__ == "__main__":
    main()

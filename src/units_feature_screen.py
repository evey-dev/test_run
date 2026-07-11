"""Discover and confirm compact force-selective TopK SAE feature panels.

The primary estimand compares two final-token swaps into the same unseen energy
target: a force-source feature swap and a matched mass-source feature swap. A
positive force-minus-mass difference supports force-specific transfer rather
than a generic response to patching any different prompt.
"""

from __future__ import annotations

import argparse
import csv
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
    condition_from_logits,
    condition_from_result,
    first_token_id,
    load_domain_saes,
    suppress_output,
)
from src.intervention import get_baseline_predictions, run_swap_in_intervention
from src.model_loader import load_model_and_tokenizer


Feature = Tuple[int, int]
CONTEXT_POOL_VERSION = 4
PROTOCOL_VERSION = 3

SYSTEMS = [
    "bridge suspension cable",
    "elevator hoist",
    "robotic arm joint",
    "aircraft landing gear",
    "railway coupling",
    "hydraulic press",
    "wind-turbine blade",
    "satellite thruster",
    "laboratory spring",
    "crane support line",
    "vehicle brake pad",
    "ship tow cable",
    "dam spillway gate",
    "conveyor drive belt",
    "rocket mounting bracket",
    "bicycle chain",
    "prosthetic knee joint",
    "industrial clamp",
    "magnetic levitation rig",
    "seismic test platform",
    "centrifuge rotor",
    "drone lifting cable",
    "submarine control surface",
    "solar-panel hinge",
    "offshore platform brace",
    "telescope positioning motor",
    "rail-switch actuator",
    "excavator boom",
    "hospital bed lift",
    "factory press ram",
    "parachute suspension line",
    "pipeline valve stem",
    "traction winch",
    "load-cell test stand",
    "aircraft control linkage",
    "robotic gripper",
    "suspension damper",
    "cargo lift",
    "cable-stayed mast",
    "pressure-vessel latch",
    "industrial gearbox",
    "steering rack",
    "turbine shaft coupling",
    "climbing safety rope",
    "machine-tool spindle",
    "gantry crane trolley",
    "shock absorber",
    "electric actuator",
    "dockside mooring line",
    "antenna positioning drive",
    "mining hoist",
    "freight elevator cable",
    "retaining-wall anchor",
    "rotating flywheel mount",
    "warehouse conveyor roller",
    "laboratory test frame",
    "aircraft flap actuator",
    "marine winch",
    "railway brake linkage",
    "telescopic boom",
    "structural tie rod",
    "automated press fixture",
    "vibration isolation mount",
    "rescue harness line",
]

OPERATING_CONTEXTS = [
    "during a controlled load test",
    "under steady operating conditions",
    "during a brief acceleration",
    "while resisting an applied displacement",
]


def checkpoint_payload(payload: Dict[str, Any], output_path: Path) -> None:
    """Atomically replace a checkpoint so interrupted Drive runs are resumable."""
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


def prompt_for(quantity: str, context: str) -> str:
    return (
        f"Fact: In an experiment involving a {context}, the official SI unit used to measure "
        f"{quantity} is named \""
    )


def generated_context_cases(seed: int) -> List[Dict[str, Any]]:
    cases = []
    for system in SYSTEMS:
        for operating_context in OPERATING_CONTEXTS:
            context = f"{system} {operating_context}"
            cases.append(
                {
                    "context": context,
                    "system": system,
                    "operating_context": operating_context,
                    "force_prompt": prompt_for("force", context),
                    "mass_prompt": prompt_for("mass", context),
                    "energy_prompt": prompt_for("energy", context),
                    "exact_prompts_absent_from_sae_corpus": True,
                }
            )
    np.random.default_rng(seed).shuffle(cases)
    return cases


def baseline_qualification(
    target_clean: Dict[str, Any],
    force_clean: Dict[str, Any],
    mass_top_id: int,
    mass_expected_ids: Iterable[int],
) -> Dict[str, Any]:
    """Qualify the causal target/source while retaining mass as a diagnostic control."""
    energy_target_correct = bool(target_clean["top_is_second"])
    force_source_correct = bool(force_clean["top_is_first"])
    mass_control_correct = int(mass_top_id) in {int(value) for value in mass_expected_ids}
    reasons = []
    if not energy_target_correct:
        reasons.append("energy target did not predict the joules prefix")
    if not force_source_correct:
        reasons.append("force source did not predict the newtons prefix")
    return {
        "eligible": energy_target_correct and force_source_correct,
        "energy_target_top_is_expected": energy_target_correct,
        "force_source_top_is_expected": force_source_correct,
        "mass_control_top_is_expected": mass_control_correct,
        "ineligible_reason": "; ".join(reasons) if reasons else None,
    }


def select_eligible_cases_by_system(
    prepared: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Choose one eligible variant per system, preferring a competent mass control."""
    selected: Dict[str, Dict[str, Any]] = {}
    for row in prepared:
        if not row.get("eligible"):
            continue
        existing = selected.get(row["system"])
        if existing is None or (
            row.get("mass_control_top_is_expected", False)
            and not existing.get("mass_control_top_is_expected", False)
        ):
            selected[row["system"]] = row
    return selected


def prepare_case(model, tokenizer, case: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(case)
    newton_id = first_token_id(tokenizer, "newtons")
    joule_id = first_token_id(tokenizer, "joules")
    kilogram_id = first_token_id(tokenizer, "kilograms")
    mass_expected_ids = {
        first_token_id(tokenizer, answer)
        for answer in ("kilograms", "kilogram", "kg")
    }
    newton_token = tokenizer.decode([newton_id])
    joule_token = tokenizer.decode([joule_id])
    kilogram_token = tokenizer.decode([kilogram_id])

    target_clean = baseline_condition(
        model,
        tokenizer,
        case["energy_prompt"],
        newton_id,
        joule_id,
    )
    force_clean = baseline_condition(
        model,
        tokenizer,
        case["force_prompt"],
        newton_id,
        joule_id,
    )
    mass_logits, mass_top_id, mass_top_token = get_baseline_predictions(
        model, tokenizer, case["mass_prompt"]
    )
    mass_clean = condition_from_logits(
        mass_logits,
        mass_top_id,
        mass_top_token,
        kilogram_id,
        newton_id,
    )
    qualification = baseline_qualification(
        target_clean,
        force_clean,
        mass_top_id,
        mass_expected_ids,
    )
    row.update(
        {
            "newton_id": newton_id,
            "joule_id": joule_id,
            "kilogram_id": kilogram_id,
            "newton_token": newton_token,
            "joule_token": joule_token,
            "kilogram_token": kilogram_token,
            "target_clean": target_clean,
            "force_clean": force_clean,
            "mass_clean": mass_clean,
            "mass_expected_token_ids": sorted(mass_expected_ids),
            **qualification,
        }
    )
    return row


def public_case_record(case: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "context",
        "system",
        "operating_context",
        "force_prompt",
        "mass_prompt",
        "energy_prompt",
        "exact_prompts_absent_from_sae_corpus",
        "eligible",
        "energy_target_top_is_expected",
        "force_source_top_is_expected",
        "mass_control_top_is_expected",
        "ineligible_reason",
        "target_clean",
        "force_clean",
        "mass_clean",
        "mass_expected_token_ids",
    ]
    return {key: case[key] for key in keep if key in case}


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
    rows = []
    for case in cases:
        with suppress_output(not verbose):
            force_result = run_swap_in_intervention(
                model,
                tokenizer,
                case["force_prompt"],
                case["energy_prompt"],
                layers,
                saes,
                features,
                [case["newton_token"], case["joule_token"]],
                position_spec=position_spec,
            )
            mass_result = run_swap_in_intervention(
                model,
                tokenizer,
                case["mass_prompt"],
                case["energy_prompt"],
                layers,
                saes,
                features,
                [case["newton_token"], case["joule_token"]],
                position_spec=position_spec,
            )

        force = condition_from_result(
            force_result,
            case["newton_token"],
            case["joule_token"],
            case["newton_id"],
            case["joule_id"],
        )
        mass = condition_from_result(
            mass_result,
            case["newton_token"],
            case["joule_token"],
            case["newton_id"],
            case["joule_id"],
        )
        clean_gap = float(case["target_clean"]["gap"])
        force_delta = float(force["gap"] - clean_gap)
        mass_delta = float(mass["gap"] - clean_gap)
        rows.append(
            {
                "context": case["context"],
                "force_source_delta": force_delta,
                "mass_source_delta": mass_delta,
                "force_minus_mass_difference": force_delta - mass_delta,
                "force_top_prediction_transfer": bool(force["top_is_first"]),
                "mass_top_prediction_transfer": bool(mass["top_is_first"]),
            }
        )
    return rows


def evaluate_broad_control(
    model,
    tokenizer,
    saes,
    cases: Sequence[Dict[str, Any]],
    position_spec: str,
    raw_mlp_swap: bool,
    verbose: bool,
) -> List[Dict[str, Any]]:
    rows = []
    for case in cases:
        with suppress_output(not verbose):
            force_result = run_swap_in_intervention(
                model,
                tokenizer,
                case["force_prompt"],
                case["energy_prompt"],
                LAYERS,
                saes,
                None,
                [case["newton_token"], case["joule_token"]],
                raw_mlp_swap=raw_mlp_swap,
                position_spec=position_spec,
            )
            mass_result = run_swap_in_intervention(
                model,
                tokenizer,
                case["mass_prompt"],
                case["energy_prompt"],
                LAYERS,
                saes,
                None,
                [case["newton_token"], case["joule_token"]],
                raw_mlp_swap=raw_mlp_swap,
                position_spec=position_spec,
            )
        force = condition_from_result(
            force_result,
            case["newton_token"],
            case["joule_token"],
            case["newton_id"],
            case["joule_id"],
        )
        mass = condition_from_result(
            mass_result,
            case["newton_token"],
            case["joule_token"],
            case["newton_id"],
            case["joule_id"],
        )
        clean_gap = float(case["target_clean"]["gap"])
        force_delta = float(force["gap"] - clean_gap)
        mass_delta = float(mass["gap"] - clean_gap)
        rows.append(
            {
                "context": case["context"],
                "force_source_delta": force_delta,
                "mass_source_delta": mass_delta,
                "force_minus_mass_difference": force_delta - mass_delta,
                "force_top_prediction_transfer": bool(force["top_is_first"]),
                "mass_top_prediction_transfer": bool(mass["top_is_first"]),
            }
        )
    return rows


def summarise_rows(rows: Sequence[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    if not rows:
        return {"eligible_cases": 0}
    force = np.asarray([row["force_source_delta"] for row in rows], dtype=float)
    mass = np.asarray([row["mass_source_delta"] for row in rows], dtype=float)
    paired = force - mass
    return {
        "eligible_cases": len(rows),
        "mean_force_source_delta": float(force.mean()),
        "bootstrap_95_ci_mean_force_source_delta": list(bootstrap_mean_ci(force, seed)),
        "mean_mass_source_delta": float(mass.mean()),
        "bootstrap_95_ci_mean_mass_source_delta": list(bootstrap_mean_ci(mass, seed + 1)),
        "mean_force_minus_mass_difference": float(paired.mean()),
        "bootstrap_95_ci_mean_force_minus_mass_difference": list(
            bootstrap_mean_ci(paired, seed + 2)
        ),
        "fraction_force_delta_positive": float(np.mean(force > 0)),
        "fraction_force_more_positive_than_mass": float(np.mean(force > mass)),
        "force_top_prediction_transfer_fraction": float(
            np.mean([row["force_top_prediction_transfer"] for row in rows])
        ),
        "mass_top_prediction_transfer_fraction": float(
            np.mean([row["mass_top_prediction_transfer"] for row in rows])
        ),
    }


def rank_feature_results(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    supportive = [row for row in results if row["summary"]["mean_force_source_delta"] > 0]
    remaining = [row for row in results if row not in supportive]

    def key(row: Dict[str, Any]) -> Tuple[float, float, float, int, int]:
        summary = row["summary"]
        return (
            -float(summary["mean_force_minus_mass_difference"]),
            -float(summary["mean_force_source_delta"]),
            -float(row["graph_attribution"]),
            int(row["layer"]),
            int(row["feature"]),
        )

    return sorted(supportive, key=key) + sorted(remaining, key=key)


def matched_random_panel(
    candidate_features: Sequence[Feature],
    reference_panel: Sequence[Feature],
    rng: np.random.Generator,
) -> List[Feature]:
    by_layer: Dict[int, List[int]] = {}
    for layer, feature in candidate_features:
        by_layer.setdefault(layer, []).append(feature)
    requested = Counter(layer for layer, _ in reference_panel)
    panel = []
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
        name = f"top_{size}"
        if size == primary_panel_size:
            name += "_primary"
        panels.append(
            {
                "name": name,
                "kind": "discovery_ranked_prefix",
                "features": list(ranked_features[: min(size, len(ranked_features))]),
            }
        )
    primary = list(ranked_features[: min(primary_panel_size, len(ranked_features))])
    panels.extend(
        [
            {
                "name": "all_positive_graph",
                "kind": "full_graph_comparator",
                "features": list(candidate_features),
            },
            {
                "name": f"bottom_{primary_panel_size}",
                "kind": "reverse_rank_control",
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
                "features": layer_features,
            }
        )
    rng = np.random.default_rng(seed)
    for index in range(random_panels):
        panels.append(
            {
                "name": f"random_matched_{index + 1:02d}",
                "kind": "layer_count_matched_random_control",
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
        "protocol_version": PROTOCOL_VERSION,
        "context_pool_version": CONTEXT_POOL_VERSION,
        "one_prompt_per_physical_system": True,
        "eligibility_rule": "correct energy target and force source; mass correctness diagnostic only",
        "model_config": str(args.model_config),
        "sae_config": str(args.sae_config),
        "graph": str(args.graph),
        "positions": args.positions,
        "seed": args.seed,
        "discovery_cases": args.discovery_cases,
        "confirmation_cases": args.confirmation_cases,
        "panel_sizes": list(args.panel_sizes),
        "primary_panel_size": args.primary_panel_size,
        "random_panels": args.random_panels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Force-selective TopK SAE discovery and confirmation")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--sae-config", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--positions", default="last")
    parser.add_argument("--discovery-cases", type=int, default=8)
    parser.add_argument("--confirmation-cases", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2787)
    parser.add_argument("--panel-sizes", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    parser.add_argument("--primary-panel-size", type=int, default=10)
    parser.add_argument("--random-panels", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/topk_units_retrain/units_topk_feature_screen.json",
    )
    args = parser.parse_args()

    if args.positions.lower() not in {"last", "final"}:
        raise ValueError("The units confirmatory screen must use --positions last")
    if args.primary_panel_size not in args.panel_sizes:
        raise ValueError("--primary-panel-size must also appear in --panel-sizes")

    repo_root = get_repo_root()
    graph_path = resolve_path(args.graph, repo_root)
    config_path = resolve_path(args.sae_config, repo_root)
    output_path = resolve_path(args.output, repo_root)
    signature = protocol_signature(args)
    started = time.time()

    payload = None
    if output_path.exists() and not args.overwrite:
        with output_path.open("r", encoding="utf-8") as handle:
            existing_payload = json.load(handle)
        if existing_payload.get("protocol_signature") != signature:
            has_intervention_results = bool(
                existing_payload.get("discovery", {}).get("feature_results")
                or existing_payload.get("confirmation", {}).get("panels")
                or existing_payload.get("confirmation", {}).get("broad_controls")
            )
            if not has_intervention_results:
                print(
                    "Replacing a pre-intervention checkpoint from an earlier screen protocol: "
                    f"{output_path}"
                )
            else:
                raise ValueError(
                    f"Existing partial output {output_path} uses a different protocol. "
                    "Use --overwrite or a new output path."
                )
        else:
            payload = existing_payload
            print(f"Resuming partial output: {output_path}")
    if payload is None:
        payload = {
            "status": "initialising",
            "protocol_signature": signature,
            "method": {
                "candidate_definition": "positive-attribution features from one force graph",
                "primary_gap": "logit(newtons prefix) minus logit(joules prefix) on energy targets",
                "screening_estimand": "force-source swap delta minus matched mass-source swap delta",
                "desired_direction": "positive",
                "selection_data": "discovery split only",
                "confirmation_data_used_for_ranking": False,
                "primary_confirmation_panel": f"top_{args.primary_panel_size}_primary",
                "primary_success_rule": (
                    "force-source mean and force-minus-mass mean are positive, with both bootstrap "
                    "95% confidence intervals wholly above zero"
                ),
                "baseline_eligibility": (
                    "The clean energy target must predict the joules prefix and the clean force "
                    "source must predict the newtons prefix. Mass-source correctness is reported "
                    "and preferred during prompt-variant selection, but does not gate eligibility."
                ),
                "scope": "final-token, error-preserving SAE feature swaps",
            },
            "discovery": {"feature_results": []},
            "confirmation": {"panels": [], "broad_controls": []},
        }
        checkpoint_payload(payload, output_path)

    graph_records = load_graph_feature_records(graph_path, LAYERS, sign="positive")
    candidate_features = [(row["layer"], row["feature"]) for row in graph_records]
    payload["candidate_features"] = graph_records
    payload["candidate_feature_count"] = len(graph_records)
    payload["candidate_layer_counts"] = dict(Counter(layer for layer, _ in candidate_features))

    case_pool = generated_context_cases(args.seed)
    with (repo_root / "data/units_data.csv").open("r", encoding="utf-8", newline="") as handle:
        sae_corpus_prompts = {row["sentence"] for row in csv.DictReader(handle)}
    prompt_overlap = [
        prompt
        for case in case_pool
        for prompt in (case["force_prompt"], case["mass_prompt"], case["energy_prompt"])
        if prompt in sae_corpus_prompts
    ]
    if prompt_overlap:
        raise ValueError("Fresh-context screen unexpectedly overlaps the SAE prompt corpus")
    print("Loading model and selected units TopK SAEs...")
    model, tokenizer, _ = load_model_and_tokenizer(repo_root / args.model_config)
    saes = load_domain_saes(model, config_path)

    prepared = []
    for index, case in enumerate(case_pool, start=1):
        print(f"[baseline {index:02d}/{len(case_pool)}] {case['context']}")
        prepared.append(prepare_case(model, tokenizer, case))
    eligible_by_system = select_eligible_cases_by_system(prepared)
    qualification_summary = {
        "screened_prompt_variants": len(prepared),
        "physical_systems_in_pool": len(SYSTEMS),
        "variants_with_expected_energy_target": sum(
            bool(row["energy_target_top_is_expected"]) for row in prepared
        ),
        "variants_with_expected_force_source": sum(
            bool(row["force_source_top_is_expected"]) for row in prepared
        ),
        "variants_with_expected_mass_control": sum(
            bool(row["mass_control_top_is_expected"]) for row in prepared
        ),
        "eligible_variants": sum(bool(row["eligible"]) for row in prepared),
        "systems_with_eligible_variant": len(eligible_by_system),
        "selected_system_variants_with_expected_mass_control": sum(
            bool(row["mass_control_top_is_expected"]) for row in eligible_by_system.values()
        ),
    }
    print("Baseline qualification summary:")
    for key, value in qualification_summary.items():
        print(f"  {key}: {value}")
    payload["baseline_audit"] = {
        "summary": qualification_summary,
        "completed_before_any_feature_intervention": True,
    }
    checkpoint_payload(payload, output_path)
    system_order = list(SYSTEMS)
    np.random.default_rng(args.seed + 1).shuffle(system_order)
    eligible_system_order = [system for system in system_order if system in eligible_by_system]
    required_systems = args.discovery_cases + args.confirmation_cases
    if len(eligible_system_order) < required_systems:
        raise ValueError(
            f"Only {len(eligible_system_order)} physical systems supplied a prompt with both a "
            f"correct clean energy target and force source; {required_systems} are required. "
            f"Qualification counts: {qualification_summary}"
        )
    discovery_systems = eligible_system_order[: args.discovery_cases]
    confirmation_systems = eligible_system_order[
        args.discovery_cases : args.discovery_cases + args.confirmation_cases
    ]
    discovery_cases = [eligible_by_system[system] for system in discovery_systems]
    confirmation_cases = [eligible_by_system[system] for system in confirmation_systems]
    payload["case_selection"] = {
        "baseline_qualification_summary": qualification_summary,
        "baseline_screened_cases": [public_case_record(row) for row in prepared],
        "discovery_systems": discovery_systems,
        "confirmation_systems": confirmation_systems,
        "discovery_contexts": [row["context"] for row in discovery_cases],
        "confirmation_contexts": [row["context"] for row in confirmation_cases],
        "split_frozen_before_feature_interventions": True,
        "system_groups_disjoint_between_splits": True,
        "one_prompt_per_physical_system": True,
        "all_exact_prompts_absent_from_sae_corpus": True,
    }
    payload["status"] = "discovery"
    checkpoint_payload(payload, output_path)

    completed = {row["key"] for row in payload["discovery"]["feature_results"]}
    for index, feature_record in enumerate(graph_records, start=1):
        key = feature_record["key"]
        if key in completed:
            print(f"[feature {index:02d}/{len(graph_records)}] {key} already complete")
            continue
        print(f"[feature {index:02d}/{len(graph_records)}] screening {key}")
        feature = (feature_record["layer"], feature_record["feature"])
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
        result["summary"] = summarise_rows(rows, args.seed + index * 10)
        result["case_effects"] = rows
        payload["discovery"]["feature_results"].append(result)
        checkpoint_payload(payload, output_path)

    ranked_results = rank_feature_results(payload["discovery"]["feature_results"])
    for rank, row in enumerate(ranked_results, start=1):
        row["discovery_rank"] = rank
        row["passes_descriptive_stability_filter"] = bool(
            row["summary"]["mean_force_source_delta"] > 0
            and row["summary"]["mean_force_minus_mass_difference"] > 0
            and row["summary"]["fraction_force_delta_positive"] >= 0.625
            and row["summary"]["fraction_force_more_positive_than_mass"] >= 0.625
        )
    ranked_features = [(row["layer"], row["feature"]) for row in ranked_results]
    payload["discovery"]["feature_results"] = ranked_results
    payload["discovery"]["frozen_feature_order"] = [feature_key(value) for value in ranked_features]
    payload["discovery"]["ranking_rule"] = (
        "Features with positive mean force-source effect are ordered first; within groups, "
        "order is descending discovery force-minus-mass difference, then force effect, "
        "graph attribution and stable feature ID."
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
        result["summary"] = summarise_rows(rows, args.seed + 10000 + index * 10)
        result["case_effects"] = rows
        payload["confirmation"]["panels"].append(result)
        checkpoint_payload(payload, output_path)

    broad_names = {row["name"] for row in payload["confirmation"]["broad_controls"]}
    for index, (name, raw) in enumerate(
        [("full_latent_swap", False), ("raw_mlp_swap", True)],
        start=1,
    ):
        if name in broad_names:
            continue
        print(f"[broad control {index}/2] {name}")
        rows = evaluate_broad_control(
            model,
            tokenizer,
            saes,
            confirmation_cases,
            args.positions,
            raw_mlp_swap=raw,
            verbose=args.verbose,
        )
        payload["confirmation"]["broad_controls"].append(
            {
                "name": name,
                "summary": summarise_rows(rows, args.seed + 20000 + index * 10),
                "case_effects": rows,
            }
        )
        checkpoint_payload(payload, output_path)

    primary_name = f"top_{args.primary_panel_size}_primary"
    primary = next(row for row in payload["confirmation"]["panels"] if row["name"] == primary_name)
    primary_summary = primary["summary"]
    force_ci = primary_summary["bootstrap_95_ci_mean_force_source_delta"]
    paired_ci = primary_summary["bootstrap_95_ci_mean_force_minus_mass_difference"]
    primary_success = bool(
        primary_summary["mean_force_source_delta"] > 0
        and force_ci[0] > 0
        and primary_summary["mean_force_minus_mass_difference"] > 0
        and paired_ci[0] > 0
    )
    random_effects = [
        row["summary"]["mean_force_minus_mass_difference"]
        for row in payload["confirmation"]["panels"]
        if row["kind"] == "layer_count_matched_random_control"
    ]
    payload["confirmation"]["primary_result"] = {
        "panel": primary_name,
        "supports_force_selectivity_under_predeclared_rule": primary_success,
        "summary": primary_summary,
        "random_control_mean_force_minus_mass_differences": random_effects,
        "primary_greater_than_random_control_fraction": (
            float(np.mean(primary_summary["mean_force_minus_mass_difference"] > np.asarray(random_effects)))
            if random_effects
            else None
        ),
    }
    payload["status"] = "complete"
    payload["runtime_seconds"] = time.time() - started
    checkpoint_payload(payload, output_path)

    print("\nPrimary confirmation result")
    print(f"  panel: {primary_name}")
    print(f"  force-source mean delta: {primary_summary['mean_force_source_delta']:+.4f}")
    print(f"  mass-source mean delta: {primary_summary['mean_mass_source_delta']:+.4f}")
    print(
        "  force-minus-mass mean: "
        f"{primary_summary['mean_force_minus_mass_difference']:+.4f} "
        f"(95% CI [{paired_ci[0]:+.4f}, {paired_ci[1]:+.4f}])"
    )
    print(f"  predeclared success rule met: {primary_success}")
    print(f"Saved screen to {output_path}")


if __name__ == "__main__":
    main()

"""Independently replicate the exploratory balanced-carry Top-20 panel.

The source experiment selected and evaluated several cumulative panel sizes.
Its predeclared Top-10 primary failed, while the secondary Top-20 panel showed a
selective causal effect. This script treats that observation as hypothesis
generation, freezes the exact twenty feature IDs, and evaluates only that panel
on intervention-untouched cases from the original baseline pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.data_utils import get_repo_root, resolve_path
from src.heldout_validation import load_domain_saes
from src.math_carry_balanced_localization import (
    checkpoint_payload,
    evaluate_causal_panel,
    feature_key,
    pair_balance,
    summarise_causal_rows,
)
from src.model_loader import load_model_and_tokenizer


Feature = Tuple[int, int]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_balanced_subset(
    cases: Sequence[Dict[str, Any]],
    count: int,
    seed: int,
    trials: int = 4000,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Choose cases using output labels only, never intervention outcomes."""
    if len(cases) < count:
        raise ValueError(f"Only {len(cases)} untouched eligible cases remain; {count} are required")
    rng = np.random.default_rng(seed)
    best_cases = None
    best_balance = None
    best_score = None
    for _ in range(trials):
        indices = rng.choice(len(cases), size=count, replace=False)
        selected = [cases[int(index)] for index in indices]
        balance = pair_balance(selected)
        per_digit_minimum = min(
            (
                min(values["carry"], values["no_carry"])
                for values in balance["counts"].values()
                if values["carry"] and values["no_carry"]
            ),
            default=0,
        )
        score = (
            balance["common_digit_count"],
            balance["matched_observations_per_class"],
            per_digit_minimum,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_cases = selected
            best_balance = balance

    assert best_cases is not None and best_balance is not None
    if best_balance["common_digit_count"] < 5:
        raise ValueError("Replication subset has fewer than five shared output-digit strata")
    best_cases.sort(key=lambda case: str(case["case_key"]))
    return best_cases, {
        "optimisation_trials": trials,
        "selection_used_only_output_digits_and_prior_baseline_eligibility": True,
        "balance": best_balance,
    }


def replication_success(summary: Dict[str, Any]) -> bool:
    paired_ci = summary["bootstrap_95_ci_mean_paired_difference"]
    return bool(
        summary["mean_carry_target_delta"] < 0
        and summary["mean_paired_difference"] < 0
        and paired_ci[1] < 0
    )


def extract_frozen_panel(source: Dict[str, Any]) -> Tuple[List[Feature], Dict[str, Any]]:
    frozen = source["sae_feature_discovery"]["frozen_top_20"]
    if len(frozen) != 20:
        raise ValueError(f"Expected exactly 20 frozen features, found {len(frozen)}")
    features = [(int(row["layer"]), int(row["feature"])) for row in frozen]
    source_panel = next(
        panel
        for panel in source["causal_confirmation"]["panels"]
        if panel["name"] == "top_20"
    )
    source_keys = [feature_key(feature) for feature in features]
    panel_keys = [row["key"] for row in source_panel["features"]]
    if source_keys != panel_keys:
        raise ValueError("Frozen Top-20 order does not match the evaluated source panel")
    return features, source_panel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replicate the frozen secondary Top-20 carry panel on untouched pairs"
    )
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--sae-config", default="configs/sae_math_topk256_config.yaml")
    parser.add_argument(
        "--source-result",
        default=(
            "outputs/math_carry_localization/"
            "math_topk256_balanced_carry_localization.json"
        ),
    )
    parser.add_argument("--replication-pairs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9787)
    parser.add_argument("--selection-trials", type=int, default=4000)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output",
        default=(
            "outputs/math_carry_localization/"
            "math_topk256_balanced_top20_replication.json"
        ),
    )
    args = parser.parse_args()

    repo_root = get_repo_root()
    source_path = resolve_path(args.source_result, repo_root)
    output_path = resolve_path(args.output, repo_root)
    if not source_path.exists():
        raise FileNotFoundError(f"Completed balanced-localisation result not found: {source_path}")
    with source_path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    if source.get("status") != "complete":
        raise ValueError("The source balanced-localisation experiment is not complete")

    features, exploratory_panel = extract_frozen_panel(source)
    exploratory_summary = exploratory_panel["causal_summary"]
    exploratory_ci = exploratory_summary["bootstrap_95_ci_mean_paired_difference"]
    if exploratory_summary["mean_paired_difference"] >= 0 or exploratory_ci[1] >= 0:
        raise ValueError(
            "The source Top-20 panel did not show the negative secondary effect this replication tests"
        )

    used_keys = set(source["split"]["discovery_case_keys"])
    used_keys.update(source["split"]["confirmation_case_keys"])
    remaining = [
        case
        for case in source["baseline_screen"]["cases"]
        if case.get("eligible") and case["case_key"] not in used_keys
    ]
    selected_cases, selection = choose_balanced_subset(
        remaining,
        args.replication_pairs,
        args.seed,
        args.selection_trials,
    )

    feature_keys = [feature_key(feature) for feature in features]
    signature = {
        "protocol_version": 1,
        "model_config": str(args.model_config),
        "sae_config": str(args.sae_config),
        "source_result_sha256": file_sha256(source_path),
        "frozen_feature_keys": feature_keys,
        "replication_pairs": args.replication_pairs,
        "replication_case_keys": [case["case_key"] for case in selected_cases],
        "seed": args.seed,
        "success_rule": (
            "negative mean carry-target delta and carry-minus-control bootstrap 95% CI "
            "wholly below zero"
        ),
    }

    if output_path.exists() and not args.overwrite:
        with output_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("protocol_signature") != signature:
            raise ValueError(
                f"Existing output {output_path} uses another protocol; use --overwrite or a new path"
            )
        if existing.get("status") == "complete":
            print(f"Completed replication already exists: {output_path}")
            print(json.dumps(existing["primary_result"], indent=2))
            return

    payload = {
        "status": "initialising",
        "protocol_signature": signature,
        "method": {
            "status_of_source_top20": "secondary hypothesis-generating result",
            "feature_selection": "exact source Top-20 frozen without reranking",
            "case_selection": (
                "intervention-untouched source baseline cases; balanced using output labels only"
            ),
            "intervention": "final-token error-preserving inhibition",
            "primary_outcome": "carry-target gap delta minus matched no-carry gap delta",
            "confirmation_data_used_for_feature_or_panel_selection": False,
            "stopping_rule": "one panel, one replication; no further panel search",
        },
        "source_result": {
            "path": str(source_path),
            "sha256": signature["source_result_sha256"],
            "source_primary_top10_passed": source["primary_result"][
                "supports_compact_carry_selectivity_under_predeclared_rule"
            ],
            "exploratory_top20_causal_summary": exploratory_summary,
        },
        "frozen_panel": [
            {"rank": rank, "layer": layer, "feature": feature, "key": feature_key((layer, feature))}
            for rank, (layer, feature) in enumerate(features, start=1)
        ],
        "case_selection": {
            "source_eligible_pairs": source["baseline_screen"]["eligible_pairs"],
            "previously_intervened_pair_count": len(used_keys),
            "intervention_untouched_eligible_pair_count": len(remaining),
            "selected_replication_pair_count": len(selected_cases),
            "selected_case_keys": [case["case_key"] for case in selected_cases],
            "selection": selection,
        },
    }
    checkpoint_payload(payload, output_path)

    started = time.time()
    print("Loading model and selected mathematics TopK-256 SAEs...")
    model, tokenizer, _ = load_model_and_tokenizer(repo_root / args.model_config)
    saes = load_domain_saes(model, resolve_path(args.sae_config, repo_root))
    print(
        f"Replicating the frozen Top-20 panel on {len(selected_cases)} of "
        f"{len(remaining)} intervention-untouched eligible pairs..."
    )
    rows = evaluate_causal_panel(
        model,
        tokenizer,
        saes,
        features,
        selected_cases,
        args.verbose,
    )
    summary = summarise_causal_rows(rows, args.seed + 1000)
    success = replication_success(summary)
    paired_ci = summary["bootstrap_95_ci_mean_paired_difference"]

    payload["replication"] = {
        "causal_summary": summary,
        "case_effects": rows,
    }
    payload["primary_result"] = {
        "replicates_frozen_top20_carry_selectivity": success,
        "causal_summary": summary,
        "interpretation_gate": (
            "The previously secondary Top-20 effect replicated on untouched cases."
            if success
            else "The previously secondary Top-20 effect did not replicate; stop panel search."
        ),
    }
    payload["status"] = "complete"
    payload["runtime_seconds"] = time.time() - started
    checkpoint_payload(payload, output_path)

    print("\nFrozen Top-20 independent replication")
    print(f"  carry-target mean delta: {summary['mean_carry_target_delta']:+.4f}")
    print(f"  no-carry mean delta: {summary['mean_no_carry_control_delta']:+.4f}")
    print(
        "  carry-minus-control: "
        f"{summary['mean_paired_difference']:+.4f} "
        f"(95% CI [{paired_ci[0]:+.4f}, {paired_ci[1]:+.4f}])"
    )
    print(f"  replication criterion met: {success}")
    print(f"Saved replication to {output_path}")


if __name__ == "__main__":
    main()

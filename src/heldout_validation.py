"""Evaluate whether graph-derived interventions generalise beyond one prompt.

No attribution graphs are regenerated. The arithmetic benchmark applies the
58+83 carry graph's positive features to unseen carry problems. The units
benchmark transfers the force representation into matched energy prompts drawn
from different objects/templates. Broad latent and raw-MLP patches are retained
as upper-bound controls.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import io
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from src.config_utils import load_yaml_config
from src.data_utils import get_repo_root, resolve_path
from src.intervention import (
    get_baseline_predictions,
    load_sae_models,
    run_inhibition_intervention,
    run_swap_in_intervention,
)
from src.model_loader import load_model_and_tokenizer


LAYERS = [4, 8, 12, 16, 20, 24, 28]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_graph_features(path: Path, layers: Sequence[int], sign: str = "positive") -> Dict[int, List[int]]:
    with path.open("r", encoding="utf-8") as handle:
        graph = json.load(handle)

    selected: Dict[int, List[int]] = {}
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
        selected.setdefault(layer, []).append(int(match.group(2)))

    if not selected:
        raise ValueError(f"No {sign} SAE feature nodes found in {path}")
    return selected


def first_token_id(tokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Text {text!r} did not produce a token")
    return int(token_ids[0])


def token_metrics(logits: torch.Tensor, token_id: int) -> Dict[str, float]:
    probabilities = torch.softmax(logits, dim=-1)
    return {
        "logit": float(logits[token_id].item()),
        "probability": float(probabilities[token_id].item()),
    }


def condition_from_logits(
    logits: torch.Tensor,
    top_id: int,
    top_token: str,
    first_id: int,
    second_id: int,
) -> Dict[str, Any]:
    first = token_metrics(logits, first_id)
    second = token_metrics(logits, second_id)
    return {
        "first": first,
        "second": second,
        "gap": first["logit"] - second["logit"],
        "top_token": top_token,
        "top_token_id": int(top_id),
        "top_is_first": bool(top_id == first_id),
        "top_is_second": bool(top_id == second_id),
    }


def condition_from_result(
    result: Dict[str, Any],
    first_token: str,
    second_token: str,
    first_id: int,
    second_id: int,
) -> Dict[str, Any]:
    targets = result["targets"]
    if first_token not in targets or second_token not in targets:
        raise KeyError(
            f"Intervention result did not contain requested targets {first_token!r}, "
            f"{second_token!r}; available targets: {list(targets)}"
        )
    first = targets[first_token]
    second = targets[second_token]
    top_token = result["top_token"]
    return {
        "first": {"logit": float(first["logit"]), "probability": float(first["prob"])},
        "second": {"logit": float(second["logit"]), "probability": float(second["prob"])},
        "gap": float(first["logit"] - second["logit"]),
        "top_token": top_token,
        "top_is_first": top_token == first_token,
        "top_is_second": top_token == second_token,
        "first_token_id": first_id,
        "second_token_id": second_id,
    }


@contextlib.contextmanager
def suppress_output(enabled: bool) -> Iterable[None]:
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def generated_math_cases(count: int, addition_csv: Path, seed: int) -> List[Dict[str, Any]]:
    training_rows = read_csv(addition_csv)
    corpus_pairs = {
        (int(row["Operand1"]), int(row["Operand2"]))
        for row in training_rows
    }
    candidates: List[Dict[str, Any]] = []

    ones_options = [(8, 3), (7, 5), (9, 4), (6, 7), (8, 5), (9, 2)]
    for tens_a in range(2, 10):
        for tens_b in range(2, 10):
            if not 10 <= tens_a + tens_b <= 17:
                continue
            for target_ones, b_ones in ones_options:
                if target_ones + b_ones < 10:
                    continue
                source_ones = 9 - b_ones
                target_a = 10 * tens_a + target_ones
                source_a = 10 * tens_a + source_ones
                b_value = 10 * tens_b + b_ones
                if (target_a, b_value) in {(58, 83), (54, 83)}:
                    continue
                if any(
                    pair in corpus_pairs
                    for pair in (
                        (target_a, b_value),
                        (b_value, target_a),
                        (source_a, b_value),
                        (b_value, source_a),
                    )
                ):
                    continue

                target_sum = target_a + b_value
                source_sum = source_a + b_value
                if target_sum // 100 != source_sum // 100 or target_sum // 100 != 1:
                    continue
                correct_digit = str((target_sum // 10) % 10)
                dropped_digit = str((source_sum // 10) % 10)
                candidates.append(
                    {
                        "target_a": target_a,
                        "source_a": source_a,
                        "b": b_value,
                        "target_sum": target_sum,
                        "source_sum": source_sum,
                        "correct_digit": correct_digit,
                        "dropped_carry_digit": dropped_digit,
                        "target_prompt": f"Question: What is {target_a} + {b_value}? Answer: 1",
                        "source_prompt": f"Question: What is {source_a} + {b_value}? Answer: 1",
                        "absent_from_sae_corpus": True,
                    }
                )

    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} graph-held-out arithmetic pairs were available")
    return candidates[:count]


def validation_indices_from_config(config_path: Path, layer: int = 4) -> set[int]:
    cfg = load_yaml_config(config_path)
    data_dir = resolve_path(cfg["data_dir"], get_repo_root())
    split_path = data_dir / "train_val_indices_per_layer.npy"
    if not split_path.exists():
        return set()
    raw = np.load(split_path, allow_pickle=True).item()
    return {int(index) for index in raw[layer]["val"]}


def generated_unit_cases(
    count: int,
    units_csv: Path,
    units_config: Path,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = read_csv(units_csv)
    corpus_sentences = {row["sentence"] for row in rows}
    validation_indices = validation_indices_from_config(units_config)
    original = 'Fact: The official SI unit for the force of a moving engine thrust is named "'
    candidates = []
    seen_objects = set()

    order = np.arange(len(rows))
    np.random.default_rng(seed).shuffle(order)
    for index in order:
        row = rows[int(index)]
        if row.get("Quantity") != "force":
            continue
        source_prompt = row["sentence"]
        if source_prompt == original or "force" not in source_prompt:
            continue
        context_object = row.get("ContextObject", "")
        if context_object in seen_objects:
            continue
        if validation_indices and int(index) not in validation_indices:
            continue
        target_prompt = re.sub(r"\bforce\b", "energy", source_prompt, count=1)
        if target_prompt == source_prompt:
            continue
        if target_prompt in corpus_sentences:
            continue
        seen_objects.add(context_object)
        candidates.append(
            {
                "dataset_index": int(index),
                "context_object": context_object,
                "source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "source_answer": "newtons",
                "target_answer": "joules",
                "source_not_in_sae_optimisation_split": bool(validation_indices),
                "source_sae_split": "validation" if validation_indices else "unknown",
                "target_absent_from_sae_corpus": True,
            }
        )
        if len(candidates) == count:
            break

    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} distinct force contexts were available for the units benchmark"
        )
    return candidates


def baseline_condition(model, tokenizer, prompt: str, first_id: int, second_id: int) -> Dict[str, Any]:
    logits, top_id, top_token = get_baseline_predictions(model, tokenizer, prompt)
    return condition_from_logits(logits, top_id, top_token, first_id, second_id)


def evaluate_math_cases(
    model,
    tokenizer,
    saes,
    features: Dict[int, List[int]],
    cases: List[Dict[str, Any]],
    verbose: bool,
    specificity_control: bool = False,
) -> List[Dict[str, Any]]:
    rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[math {index:02d}/{len(cases)}] {case['source_a']}+{case['b']} -> {case['target_a']}+{case['b']}")
        correct_id = first_token_id(tokenizer, case["correct_digit"])
        dropped_id = first_token_id(tokenizer, case["dropped_carry_digit"])
        correct_token = tokenizer.decode([correct_id])
        dropped_token = tokenizer.decode([dropped_id])

        source_ids = tokenizer(case["source_prompt"], return_tensors="pt")["input_ids"]
        target_ids = tokenizer(case["target_prompt"], return_tensors="pt")["input_ids"]
        if source_ids.shape[1] != target_ids.shape[1]:
            case["skipped_reason"] = "source and target token lengths differ"
            rows.append(case)
            continue

        clean = baseline_condition(model, tokenizer, case["target_prompt"], correct_id, dropped_id)
        source_clean = baseline_condition(model, tokenizer, case["source_prompt"], dropped_id, correct_id)
        source_sparse_result = None
        with suppress_output(not verbose):
            sparse_result = run_inhibition_intervention(
                model,
                tokenizer,
                case["target_prompt"],
                LAYERS,
                saes,
                features,
                [correct_token, dropped_token],
                position_spec="all",
            )
            full_result = run_swap_in_intervention(
                model,
                tokenizer,
                case["source_prompt"],
                case["target_prompt"],
                LAYERS,
                saes,
                None,
                [correct_token, dropped_token],
                position_spec="all",
            )
            raw_result = run_swap_in_intervention(
                model,
                tokenizer,
                case["source_prompt"],
                case["target_prompt"],
                LAYERS,
                saes,
                None,
                [correct_token, dropped_token],
                raw_mlp_swap=True,
                position_spec="all",
            )
            if specificity_control:
                source_sparse_result = run_inhibition_intervention(
                    model,
                    tokenizer,
                    case["source_prompt"],
                    LAYERS,
                    saes,
                    features,
                    [dropped_token, correct_token],
                    position_spec="all",
                )

        row = dict(case)
        row["correct_token"] = correct_token
        row["dropped_carry_token"] = dropped_token
        row["eligible"] = bool(clean["top_is_first"] and source_clean["top_is_first"])
        row["source_clean"] = source_clean
        row["conditions"] = {
            "clean": clean,
            "sparse_inhibition": condition_from_result(
                sparse_result, correct_token, dropped_token, correct_id, dropped_id
            ),
            "full_latent_swap": condition_from_result(
                full_result, correct_token, dropped_token, correct_id, dropped_id
            ),
            "raw_mlp_swap": condition_from_result(
                raw_result, correct_token, dropped_token, correct_id, dropped_id
            ),
        }
        if source_sparse_result is not None:
            source_condition = condition_from_result(
                source_sparse_result,
                dropped_token,
                correct_token,
                dropped_id,
                correct_id,
            )
            row["specificity_control"] = {
                "definition": "same carry-graph features inhibited on the matched no-carry source",
                "clean": source_clean,
                "sparse_inhibition": source_condition,
                "gap_delta": source_condition["gap"] - source_clean["gap"],
            }
        rows.append(row)
    return rows


def evaluate_unit_cases(
    model,
    tokenizer,
    saes,
    features: Dict[int, List[int]],
    cases: List[Dict[str, Any]],
    verbose: bool,
) -> List[Dict[str, Any]]:
    rows = []
    newton_id = first_token_id(tokenizer, "newtons")
    joule_id = first_token_id(tokenizer, "joules")
    newton_token = tokenizer.decode([newton_id])
    joule_token = tokenizer.decode([joule_id])

    for index, case in enumerate(cases, start=1):
        print(f"[units {index:02d}/{len(cases)}] {case['context_object']}")
        source_ids = tokenizer(case["source_prompt"], return_tensors="pt")["input_ids"]
        target_ids = tokenizer(case["target_prompt"], return_tensors="pt")["input_ids"]
        if source_ids.shape[1] != target_ids.shape[1]:
            case["skipped_reason"] = "source and target token lengths differ"
            rows.append(case)
            continue

        # The common gap is source-answer prefix minus target-answer prefix: new - j.
        clean = baseline_condition(model, tokenizer, case["target_prompt"], newton_id, joule_id)
        source_clean = baseline_condition(model, tokenizer, case["source_prompt"], newton_id, joule_id)
        with suppress_output(not verbose):
            sparse_result = run_swap_in_intervention(
                model,
                tokenizer,
                case["source_prompt"],
                case["target_prompt"],
                LAYERS,
                saes,
                features,
                [newton_token, joule_token],
                position_spec="all",
            )
            full_result = run_swap_in_intervention(
                model,
                tokenizer,
                case["source_prompt"],
                case["target_prompt"],
                LAYERS,
                saes,
                None,
                [newton_token, joule_token],
                position_spec="all",
            )
            raw_result = run_swap_in_intervention(
                model,
                tokenizer,
                case["source_prompt"],
                case["target_prompt"],
                LAYERS,
                saes,
                None,
                [newton_token, joule_token],
                raw_mlp_swap=True,
                position_spec="all",
            )

        row = dict(case)
        row["source_prefix_token"] = newton_token
        row["target_prefix_token"] = joule_token
        row["eligible"] = bool(clean["top_is_second"] and source_clean["top_is_first"])
        row["source_clean"] = source_clean
        row["conditions"] = {
            "clean": clean,
            "sparse_feature_swap": condition_from_result(
                sparse_result, newton_token, joule_token, newton_id, joule_id
            ),
            "full_latent_swap": condition_from_result(
                full_result, newton_token, joule_token, newton_id, joule_id
            ),
            "raw_mlp_swap": condition_from_result(
                raw_result, newton_token, joule_token, newton_id, joule_id
            ),
        }
        rows.append(row)
    return rows


def bootstrap_mean_ci(values: np.ndarray, seed: int, samples: int = 5000) -> Tuple[float, float]:
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summarise(rows: List[Dict[str, Any]], desired_direction: int, seed: int) -> Dict[str, Any]:
    eligible = [row for row in rows if row.get("eligible") and "conditions" in row]
    condition_names = sorted(
        {name for row in eligible for name in row["conditions"] if name != "clean"}
    )
    summary: Dict[str, Any] = {
        "total_cases": len(rows),
        "eligible_cases": len(eligible),
        "eligibility_definition": "clean target and source both have the expected first-token top prediction",
        "desired_delta_sign": "positive" if desired_direction > 0 else "negative",
        "conditions": {},
    }
    for condition in condition_names:
        deltas = np.asarray(
            [row["conditions"][condition]["gap"] - row["conditions"]["clean"]["gap"] for row in eligible],
            dtype=float,
        )
        if deltas.size == 0:
            continue
        low, high = bootstrap_mean_ci(deltas, seed)
        top_transfer = [row["conditions"][condition]["top_is_second"] for row in eligible]
        if desired_direction > 0:
            # Units define first as source answer, so successful top transfer is first.
            top_transfer = [row["conditions"][condition]["top_is_first"] for row in eligible]
        summary["conditions"][condition] = {
            "mean_gap_delta": float(deltas.mean()),
            "median_gap_delta": float(np.median(deltas)),
            "bootstrap_95_ci_mean": [low, high],
            "fraction_in_predicted_direction": float(np.mean(desired_direction * deltas > 0)),
            "top_prediction_transfer_fraction": float(np.mean(top_transfer)),
        }
    return summary


def summarise_math_specificity(rows: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row.get("eligible")
        and "specificity_control" in row
        and "conditions" in row
    ]
    target_deltas = np.asarray(
        [
            row["conditions"]["sparse_inhibition"]["gap"]
            - row["conditions"]["clean"]["gap"]
            for row in eligible
        ],
        dtype=float,
    )
    source_deltas = np.asarray(
        [row["specificity_control"]["gap_delta"] for row in eligible],
        dtype=float,
    )
    if not eligible:
        return {"eligible_cases": 0}

    paired_difference = target_deltas - source_deltas
    target_ci = bootstrap_mean_ci(target_deltas, seed)
    source_ci = bootstrap_mean_ci(source_deltas, seed + 1)
    paired_ci = bootstrap_mean_ci(paired_difference, seed + 2)
    return {
        "eligible_cases": len(eligible),
        "target_definition": "carry-target correct-minus-dropped-carry gap delta",
        "control_definition": "matched no-carry correct-minus-carry-injected alternative gap delta",
        "paired_difference_definition": "target delta minus matched no-carry control delta; negative favours carry selectivity",
        "mean_target_delta": float(target_deltas.mean()),
        "bootstrap_95_ci_mean_target_delta": list(target_ci),
        "mean_no_carry_control_delta": float(source_deltas.mean()),
        "bootstrap_95_ci_mean_no_carry_control_delta": list(source_ci),
        "mean_paired_difference": float(paired_difference.mean()),
        "bootstrap_95_ci_mean_paired_difference": list(paired_ci),
        "fraction_target_more_negative_than_control": float(
            np.mean(target_deltas < source_deltas)
        ),
    }


def load_domain_saes(model, config_path: Path) -> Dict[int, Any]:
    cfg = load_yaml_config(config_path)
    sae_dir = resolve_path(cfg["output_dir"], get_repo_root())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_sae_models(
        LAYERS,
        sae_dir,
        int(cfg.get("hidden_size", 2560)),
        int(cfg.get("latent_dim", 8192)),
        device,
        model.dtype,
    )


def save_payload(payload: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def print_summary(label: str, summary: Dict[str, Any]) -> None:
    print(f"\n{label}: {summary['eligible_cases']}/{summary['total_cases']} baseline-qualified cases")
    for condition, metrics in summary["conditions"].items():
        ci = metrics["bootstrap_95_ci_mean"]
        print(
            f"  {condition:<24} mean delta={metrics['mean_gap_delta']:+.3f} "
            f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
            f"direction={metrics['fraction_in_predicted_direction']:.2f}  "
            f"top-transfer={metrics['top_prediction_transfer_fraction']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph-held-out intervention benchmark")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--math-sae-config", default="configs/sae_math_final_train_config.yaml")
    parser.add_argument("--units-sae-config", default="configs/sae_units_final_train_config.yaml")
    parser.add_argument("--math-graph", default="outputs/math_final_carry_58_83_4v3_graph.json")
    parser.add_argument("--units-graph", default="outputs/units_final_force_graph.json")
    parser.add_argument("--math-cases", type=int, default=12)
    parser.add_argument("--unit-cases", type=int, default=12)
    parser.add_argument("--skip-math", action="store_true")
    parser.add_argument("--skip-units", action="store_true")
    parser.add_argument(
        "--math-specificity-control",
        action="store_true",
        help="Also inhibit carry-graph features on each matched no-carry source prompt",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=787)
    parser.add_argument("--output", default="outputs/final_heldout_validation.json")
    args = parser.parse_args()

    repo_root = get_repo_root()
    output_path = resolve_path(args.output, repo_root)
    started = time.time()
    payload: Dict[str, Any] = {
        "method": {
            "seed": args.seed,
            "layers": LAYERS,
            "positions": "all",
            "graph_features": "positive-attribution nodes selected on one original graph prompt",
            "literal_edit_strength": 1.0,
            "math_specificity_control": args.math_specificity_control,
            "important_scope_note": (
                "Graph-held-out means the evaluation prompt was not used to construct the graph. "
                "The units source prompts may occur in the SAE validation corpus; target energy variants do not."
            ),
        }
    }

    print("Loading language model once for both benchmarks...")
    model, tokenizer, _ = load_model_and_tokenizer(repo_root / args.model_config)

    if not args.skip_math:
        math_graph = resolve_path(args.math_graph, repo_root)
        math_features = load_graph_features(math_graph, LAYERS, sign="positive")
        math_cases = generated_math_cases(args.math_cases, repo_root / "data/addition_data.csv", args.seed)
        print(f"Loaded {sum(map(len, math_features.values()))} positive math graph features")
        math_saes = load_domain_saes(model, repo_root / args.math_sae_config)
        math_rows = evaluate_math_cases(
            model,
            tokenizer,
            math_saes,
            math_features,
            math_cases,
            args.verbose,
            specificity_control=args.math_specificity_control,
        )
        payload["math"] = {
            "gap_definition": "logit(correct tens digit) - logit(dropped-carry digit)",
            "predicted_direction": "negative delta",
            "cases": math_rows,
            "summary": summarise(math_rows, desired_direction=-1, seed=args.seed),
        }
        if args.math_specificity_control:
            payload["math"]["specificity_summary"] = summarise_math_specificity(
                math_rows, seed=args.seed
            )
        print_summary("Arithmetic generalisation", payload["math"]["summary"])
        if args.math_specificity_control:
            specificity = payload["math"]["specificity_summary"]
            if specificity["eligible_cases"]:
                paired_ci = specificity["bootstrap_95_ci_mean_paired_difference"]
                print(
                    "\nCarry specificity control: "
                    f"target mean={specificity['mean_target_delta']:+.3f}, "
                    f"no-carry mean={specificity['mean_no_carry_control_delta']:+.3f}, "
                    f"paired difference={specificity['mean_paired_difference']:+.3f} "
                    f"(95% CI [{paired_ci[0]:+.3f}, {paired_ci[1]:+.3f}])"
                )
            else:
                print("\nCarry specificity control: no baseline-qualified cases")
        save_payload(payload, output_path)
        del math_saes
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.skip_units:
        units_config_path = repo_root / args.units_sae_config
        units_graph = resolve_path(args.units_graph, repo_root)
        unit_features = load_graph_features(units_graph, LAYERS, sign="positive")
        unit_cases = generated_unit_cases(
            args.unit_cases,
            repo_root / "data/units_data.csv",
            units_config_path,
            args.seed,
        )
        print(f"Loaded {sum(map(len, unit_features.values()))} positive units graph features")
        unit_saes = load_domain_saes(model, units_config_path)
        unit_rows = evaluate_unit_cases(model, tokenizer, unit_saes, unit_features, unit_cases, args.verbose)
        payload["units"] = {
            "gap_definition": "logit(force prefix 'new') - logit(energy prefix 'j') on energy targets",
            "predicted_direction": "positive delta",
            "cases": unit_rows,
            "summary": summarise(unit_rows, desired_direction=1, seed=args.seed),
        }
        print_summary("SI-unit generalisation", payload["units"]["summary"])
        del unit_saes

    payload["runtime_seconds"] = time.time() - started
    save_payload(payload, output_path)
    print(f"\nSaved held-out validation results to {output_path}")
    print(f"Total runtime: {payload['runtime_seconds'] / 60:.1f} minutes")


if __name__ == "__main__":
    main()

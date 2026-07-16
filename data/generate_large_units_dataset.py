"""Generate a larger, balanced SI-units corpus for a data-scale ablation.

The original units corpus contains 1,000 prompts. This generator retains every
original row and augments it with prompts from a fully crossed set of quantities,
apparatuses, operating conditions, and templates. Apparatus and condition
wording is shared across quantities so that the quantity itself, rather than a
quantity-specific object vocabulary, remains the most reliable cue to the
answer.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List


DEFAULT_SEED = 787

QUANTITIES = {
    "temperature": {"answer": "kelvin", "distractor": "joules"},
    "mass": {"answer": "kilograms", "distractor": "seconds"},
    "time": {"answer": "seconds", "distractor": "meters"},
    "force": {"answer": "newtons", "distractor": "volts"},
    "energy": {"answer": "joules", "distractor": "ohms"},
}

TRAINING_SYSTEMS = [
    "bench-scale actuator",
    "sealed test chamber",
    "instrumented cart",
    "prototype drive assembly",
    "calibration fixture",
    "composite test coupon",
    "laboratory rotor",
    "portable sensor module",
    "thermal control rig",
    "electromechanical stage",
    "fluid circulation loop",
    "vacuum test assembly",
    "precision positioning table",
    "modular pump unit",
    "small-scale turbine",
    "material testing fixture",
    "motorised lead screw",
    "pneumatic test device",
    "optical bench assembly",
    "environmental test enclosure",
    "instrumented flywheel",
    "prototype valve assembly",
    "benchtop compressor",
    "controlled reaction vessel",
    "mobile measurement platform",
]

OPERATING_CONDITIONS = [
    "at its nominal operating point",
    "during a repeatable calibration cycle",
    "under a fixed laboratory setting",
    "while the apparatus is held at steady state",
    "during a controlled measurement interval",
    "under a documented reference condition",
    "while sensors record a stable reading",
    "during a standard verification trial",
    "under a reproducible test condition",
    "while operating within its rated range",
]

TEMPLATES = [
    "Fact: The official SI unit used to report the {quantity} of a {system} {condition} is named \"",
    "Fact: A researcher measuring the {quantity} of a {system} {condition} records the result in the unit named \"",
    "Fact: For a {system} {condition}, the standard scientific unit for {quantity} is named \"",
    "Fact: During laboratory analysis of a {system} {condition}, its {quantity} is expressed in the unit named \"",
    "Fact: The metric unit used to quantify the {quantity} of a {system} {condition} is called the \"",
    "Fact: When documenting a {system} {condition}, scientists state its {quantity} in the unit named \"",
    "Fact: An SI measurement of the {quantity} of a {system} {condition} uses the unit named \"",
    "Fact: The measurement standard for the {quantity} of a {system} {condition} is the unit named \"",
]


def read_base_rows(path: Path) -> List[Dict[str, str]]:
    """Load and normalise the original units corpus retained by the ablation."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = []
    for source in source_rows:
        quantity = source["Quantity"]
        labels = QUANTITIES[quantity]
        rows.append(
            {
                "Quantity": quantity,
                "ContextObject": source.get("ContextObject", ""),
                "OperatingContext": "original corpus",
                "TemplateID": "original",
                "Answer": labels["answer"],
                "DistractorAnswer": labels["distractor"],
                "sentence": source["sentence"],
            }
        )
    return rows


def build_rows(
    count: int = 10_000,
    seed: int = DEFAULT_SEED,
    base_csv: str | Path | None = None,
) -> List[Dict[str, str]]:
    """Return a deterministic, nested, quantity-balanced prompt corpus."""
    quantity_count = len(QUANTITIES)
    base_path = Path(base_csv) if base_csv is not None else Path(__file__).with_name("units_data.csv")
    base_rows = read_base_rows(base_path)
    additional_per_quantity = len(TRAINING_SYSTEMS) * len(OPERATING_CONDITIONS) * len(TEMPLATES)
    maximum = len(base_rows) + quantity_count * additional_per_quantity
    if count < len(base_rows) or count > maximum:
        raise ValueError(f"count must lie in [{len(base_rows)}, {maximum}], got {count}")
    if count % quantity_count:
        raise ValueError(f"count must be divisible by {quantity_count} for exact quantity balance")

    rng = random.Random(seed)
    rows_per_quantity = count // quantity_count
    base_by_quantity = {
        quantity: [row for row in base_rows if row["Quantity"] == quantity]
        for quantity in QUANTITIES
    }
    if len({len(rows) for rows in base_by_quantity.values()}) != 1:
        raise ValueError("The original units corpus is not quantity-balanced")

    rows: List[Dict[str, str]] = []
    for quantity, labels in QUANTITIES.items():
        candidates: List[Dict[str, str]] = []
        for system in TRAINING_SYSTEMS:
            for condition in OPERATING_CONDITIONS:
                for template_index, template in enumerate(TEMPLATES, start=1):
                    candidates.append(
                        {
                            "Quantity": quantity,
                            "ContextObject": system,
                            "OperatingContext": condition,
                            "TemplateID": str(template_index),
                            "Answer": labels["answer"],
                            "DistractorAnswer": labels["distractor"],
                            "sentence": template.format(
                                quantity=quantity,
                                system=system,
                                condition=condition,
                            ),
                        }
                    )
        rng.shuffle(candidates)
        retained = base_by_quantity[quantity]
        additional_needed = rows_per_quantity - len(retained)
        if additional_needed < 0:
            raise ValueError("Requested count is too small to retain the full original corpus")
        retained_sentences = {row["sentence"] for row in retained}
        candidates = [row for row in candidates if row["sentence"] not in retained_sentences]
        rows.extend(retained)
        rows.extend(candidates[:additional_needed])

    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["PromptID"] = f"units-large-{index:05d}"

    sentences = [row["sentence"] for row in rows]
    quantity_counts = Counter(row["Quantity"] for row in rows)
    if len(sentences) != len(set(sentences)):
        raise AssertionError("Generated units prompts are not unique")
    if len(set(quantity_counts.values())) != 1:
        raise AssertionError(f"Generated units corpus is not balanced: {quantity_counts}")
    return rows


def write_rows(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "PromptID",
        "Quantity",
        "ContextObject",
        "OperatingContext",
        "TemplateID",
        "Answer",
        "DistractorAnswer",
        "sentence",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the larger balanced SI-units corpus")
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--base-csv", default="data/units_data.csv")
    parser.add_argument("--output", default="data/units_large_10000.csv")
    args = parser.parse_args()

    rows = build_rows(count=args.count, seed=args.seed, base_csv=args.base_csv)
    output_path = Path(args.output)
    write_rows(rows, output_path)
    counts = Counter(row["Quantity"] for row in rows)
    print(f"Saved {len(rows)} unique prompts to {output_path}")
    print(f"Retained all prompts from {args.base_csv}")
    print("Quantity counts:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()

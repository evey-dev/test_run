"""Generate a 10,000-prompt relation-balanced geography SAE corpus.

Half of the prompts ask for the capital associated with a non-capital city;
half ask for the country or state containing that city.  Every distinct prompt
from the original 1,000-row corpus is retained.  Augmentation templates differ from
the held-out relation-screen templates in ``capitals_relation_feature_screen``.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DEFAULT_SEED = 787

CAPITAL_TEMPLATES = [
    "Question: Which city is the capital of the {type} containing {city}? Answer:",
    "Fact: An atlas lists the capital of the {type} containing {city} as",
    "Fact: The governmental seat of the {type} in which {city} lies is",
    "Fact: {city} belongs to a {type} whose capital city is",
    "Fact: For the {type} that contains {city}, the administrative capital is",
    "Question: {city} lies within a {type}. What is that {type}'s capital? Answer:",
    "Fact: A political map gives the capital associated with {city}'s {type} as",
    "Fact: The principal seat of government for the {type} containing {city} is",
]

LOCATION_TEMPLATES = [
    "Question: Which {type} contains {city}? Answer:",
    "Fact: The {type} in which {city} is located is",
    "Fact: An atlas places {city} in the {type} called",
    "Fact: Geographically, {city} belongs to the {type} named",
    "Question: {city} is a city in which {type}? Answer:",
    "Fact: The political region containing {city} is the {type} named",
    "Fact: On a political map, {city} appears inside the {type} called",
    "Fact: The containing {type} associated with {city} is",
]


def read_base_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"Location", "Type", "Answer", "DistractorAnswer", "sentence"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Expected columns {sorted(required)} in {path}")
    return rows


def unique_by_sentence(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    unique: Dict[str, Dict[str, str]] = {}
    for row in rows:
        unique.setdefault(row["sentence"], row)
    return list(unique.values())


def balanced_take(
    rows: Sequence[Dict[str, str]],
    count: int,
    seed: int,
) -> List[Dict[str, str]]:
    """Select nearly equal counts per location without reusing a sentence."""
    rng = random.Random(seed)
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in unique_by_sentence(rows):
        grouped[row["Location"]].append(row)
    locations = sorted(grouped)
    rng.shuffle(locations)
    for values in grouped.values():
        rng.shuffle(values)

    selected: List[Dict[str, str]] = []
    while len(selected) < count:
        progress = False
        for location in locations:
            if grouped[location]:
                selected.append(grouped[location].pop())
                progress = True
                if len(selected) == count:
                    break
        if not progress:
            raise ValueError(f"Only {len(selected)} unique balanced candidates exist; {count} required")
        rng.shuffle(locations)
    return selected


def build_rows(
    count: int = 10_000,
    seed: int = DEFAULT_SEED,
    base_csv: str | Path = "data/capitals_data.csv",
) -> List[Dict[str, str]]:
    if count % 2:
        raise ValueError("count must be even so capital and location relations are balanced")
    base_rows = read_base_rows(Path(base_csv))
    relation_target = count // 2
    if len(base_rows) > relation_target:
        raise ValueError("The retained base corpus exceeds the capital-relation allocation")

    original_capital = []
    capital_candidates = []
    location_candidates = []
    for source in base_rows:
        common = {
            "Location": source["Location"].strip(),
            "Type": source["Type"].strip(),
            "Capital": source["Answer"].strip(),
            "ContextCity": source["DistractorAnswer"].strip(),
        }
        original_capital.append(
            {
                **common,
                "Relation": "capital",
                "TemplateID": "original",
                "Answer": common["Capital"],
                "DistractorAnswer": common["Location"],
                "Source": "retained_original",
                "sentence": source["sentence"],
            }
        )
        for index, template in enumerate(CAPITAL_TEMPLATES, start=1):
            capital_candidates.append(
                {
                    **common,
                    "Relation": "capital",
                    "TemplateID": f"capital_{index}",
                    "Answer": common["Capital"],
                    "DistractorAnswer": common["Location"],
                    "Source": "augmented",
                    "sentence": template.format(type=common["Type"], city=common["ContextCity"]),
                }
            )
        for index, template in enumerate(LOCATION_TEMPLATES, start=1):
            location_candidates.append(
                {
                    **common,
                    "Relation": "location",
                    "TemplateID": f"location_{index}",
                    "Answer": common["Location"],
                    "DistractorAnswer": common["Capital"],
                    "Source": "augmented",
                    "sentence": template.format(type=common["Type"], city=common["ContextCity"]),
                }
            )

    original_capital = unique_by_sentence(original_capital)
    retained_sentences = {row["sentence"] for row in original_capital}
    capital_candidates = [
        row for row in capital_candidates if row["sentence"] not in retained_sentences
    ]
    capital_rows = original_capital + balanced_take(
        capital_candidates,
        relation_target - len(original_capital),
        seed + 1,
    )
    location_rows = balanced_take(location_candidates, relation_target, seed + 2)
    rows = unique_by_sentence(capital_rows + location_rows)
    if len(rows) != count:
        raise AssertionError(f"Expected {count} unique prompts, constructed {len(rows)}")

    rng = random.Random(seed)
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["PromptID"] = f"capitals-large-{index:05d}"

    relation_counts = Counter(row["Relation"] for row in rows)
    if set(relation_counts.values()) != {relation_target}:
        raise AssertionError(f"Relations are not balanced: {relation_counts}")
    if not retained_sentences.issubset({row["sentence"] for row in rows}):
        raise AssertionError("Not every original capital prompt was retained")
    return rows


def write_rows(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "PromptID",
        "Relation",
        "Location",
        "Type",
        "Capital",
        "ContextCity",
        "TemplateID",
        "Answer",
        "DistractorAnswer",
        "Source",
        "sentence",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a relation-balanced capitals corpus")
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--base-csv", default="data/capitals_data.csv")
    parser.add_argument("--output", default="data/capitals_large_10000.csv")
    args = parser.parse_args()

    rows = build_rows(args.count, args.seed, args.base_csv)
    write_rows(rows, Path(args.output))
    print(f"Saved {len(rows)} unique prompts to {args.output}")
    print("Relation counts:", dict(sorted(Counter(row["Relation"] for row in rows).items())))
    print("Distinct locations:", len({row["Location"] for row in rows}))


if __name__ == "__main__":
    main()

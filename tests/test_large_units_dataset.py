from collections import Counter
import csv
from pathlib import Path

from data.generate_large_units_dataset import (
    QUANTITIES,
    TRAINING_SYSTEMS,
    build_rows,
)
from src.units_feature_screen import SYSTEMS, generated_context_cases


def test_large_units_corpus_is_balanced_unique_and_deterministic() -> None:
    rows = build_rows(count=10_000, seed=787)
    repeated = build_rows(count=10_000, seed=787)

    assert rows == repeated
    assert len(rows) == 10_000
    assert len({row["sentence"] for row in rows}) == 10_000
    assert Counter(row["Quantity"] for row in rows) == {
        quantity: 2_000 for quantity in QUANTITIES
    }
    with Path("data/units_data.csv").open("r", encoding="utf-8", newline="") as handle:
        original_sentences = {row["sentence"] for row in csv.DictReader(handle)}
    assert original_sentences.issubset({row["sentence"] for row in rows})


def test_large_units_training_and_confirmation_systems_are_disjoint() -> None:
    assert set(TRAINING_SYSTEMS).isdisjoint(SYSTEMS)

    corpus = {row["sentence"] for row in build_rows(count=10_000, seed=787)}
    confirmation = {
        prompt
        for case in generated_context_cases(seed=2787)
        for prompt in (case["force_prompt"], case["mass_prompt"], case["energy_prompt"])
    }
    assert corpus.isdisjoint(confirmation)

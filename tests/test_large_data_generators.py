import csv
from collections import Counter
from pathlib import Path

from data.generate_large_capitals_dataset import build_rows as build_capitals_rows
from data.generate_large_math_dataset import (
    EXPLICIT_HELDOUT,
    build_rows as build_math_rows,
    canonical_pair,
)
from src.heldout_validation import generated_math_cases


def test_large_capitals_corpus_is_unique_balanced_and_nested():
    base_path = Path("data/capitals_data.csv")
    rows = build_capitals_rows(count=10_000, seed=787, base_csv=base_path)

    assert len(rows) == 10_000
    assert len({row["sentence"] for row in rows}) == 10_000
    assert Counter(row["Relation"] for row in rows) == {"capital": 5_000, "location": 5_000}
    with base_path.open(encoding="utf-8") as handle:
        original_prompts = {line["sentence"] for line in csv.DictReader(handle)}
    assert original_prompts.issubset({row["sentence"] for row in rows})


def test_large_math_corpus_matches_tens_position_and_reserves_cases(tmp_path):
    rows = build_math_rows(count=10_000, seed=787)

    assert len(rows) == 10_000
    assert len({row["sentence"] for row in rows}) == 10_000
    assert Counter(row["IsCarry"] for row in rows) == {"0": 5_000, "1": 5_000}
    assert {row["TeacherForcedPrefix"] for row in rows} == {"1"}
    training_pairs = {
        canonical_pair(int(row["Operand1"]), int(row["Operand2"])) for row in rows
    }
    assert all(canonical_pair(*pair) not in training_pairs for pair in EXPLICIT_HELDOUT)

    corpus_path = tmp_path / "addition_large.csv"
    from data.generate_large_math_dataset import write_rows

    write_rows(rows, corpus_path)
    fresh = generated_math_cases(140, corpus_path, seed=4787)
    assert len(fresh) == 140
    assert all(case["absent_from_sae_corpus"] for case in fresh)

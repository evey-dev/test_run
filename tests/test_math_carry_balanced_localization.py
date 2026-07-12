import numpy as np

from src.math_carry_balanced_localization import (
    choose_balanced_split,
    collect_case_keys,
    conditional_score_summary,
    fit_conditional_direction,
    rank_carry_features,
)


def synthetic_records(repetitions: int = 8):
    records = []
    for digit in ("1", "2", "3"):
        for label in (0, 1):
            for index in range(repetitions):
                records.append(
                    {
                        "case_key": f"{digit}-{label}-{index}",
                        "output_digit": digit,
                        "carry_label": label,
                    }
                )
    return records


def test_collect_case_keys_recovers_nested_public_cases() -> None:
    payload = {
        "case_key": "24+83->28+83",
        "nested": [{"source_a": 54, "target_a": 58, "b": 83}],
    }

    assert collect_case_keys(payload) == {"24+83->28+83", "54+83->58+83"}


def test_balanced_split_is_disjoint_and_has_shared_output_strata() -> None:
    cases = []
    for index in range(80):
        dropped = str(1 + index % 6)
        correct = str(2 + index % 6)
        cases.append(
            {
                "case_key": f"case-{index}",
                "source_a": 40 + index,
                "target_a": 41 + index,
                "b": 83,
                "dropped_carry_digit": dropped,
                "correct_digit": correct,
            }
        )

    discovery, confirmation, balance = choose_balanced_split(
        cases, discovery_pairs=24, confirmation_pairs=24, seed=787, trials=200
    )

    assert {row["case_key"] for row in discovery}.isdisjoint(
        {row["case_key"] for row in confirmation}
    )
    assert balance["discovery"]["common_digit_count"] >= 5
    assert balance["confirmation"]["common_digit_count"] >= 5


def test_conditional_summary_ignores_output_digit_offset() -> None:
    records = synthetic_records()
    digit_only = np.asarray([10.0 * int(row["output_digit"]) for row in records])
    carry_signal = digit_only + np.asarray([row["carry_label"] for row in records])

    digit_summary = conditional_score_summary(digit_only, records, seed=787, bootstrap_samples=200)
    carry_summary = conditional_score_summary(carry_signal, records, seed=787, bootstrap_samples=200)

    assert np.isclose(digit_summary["mean_within_digit_carry_minus_no_carry"], 0.0)
    assert np.isclose(digit_summary["output_digit_conditioned_auc"], 0.5)
    assert np.isclose(carry_summary["mean_within_digit_carry_minus_no_carry"], 1.0)
    assert np.isclose(carry_summary["output_digit_conditioned_auc"], 1.0)


def test_feature_ranking_prefers_carry_signal_after_digit_conditioning() -> None:
    records = synthetic_records()
    rows = len(records)
    matrix = np.ones((rows, 24), dtype=np.float32)
    labels = np.asarray([row["carry_label"] for row in records], dtype=np.float32)
    digits = np.asarray([int(row["output_digit"]) for row in records], dtype=np.float32)
    matrix += digits[:, None] * 0.3
    matrix += labels[:, None] * 0.2
    matrix[:, 0] += labels * 2.0

    ranked, _, ranking_records = rank_carry_features(
        matrix,
        records,
        layers=[4, 8],
        minimum_active_fraction=0.1,
        minimum_positive_stratum_fraction=0.6,
    )

    assert ranked[0] == 0
    assert ranking_records[0]["layer"] == 4
    assert ranking_records[0]["feature"] == 0


def test_raw_direction_generalises_when_carry_is_not_the_digit_offset() -> None:
    discovery_records = synthetic_records(repetitions=6)
    confirmation_records = synthetic_records(repetitions=4)

    def matrix(records):
        digits = np.asarray([int(row["output_digit"]) for row in records], dtype=np.float32)
        labels = np.asarray([row["carry_label"] for row in records], dtype=np.float32)
        return np.column_stack([digits * 5.0, labels * 2.0, digits + labels])

    fit = fit_conditional_direction(matrix(discovery_records), discovery_records)
    confirmation_matrix = matrix(confirmation_records)
    standardised = (confirmation_matrix - fit["mean"]) / fit["scale"]
    for index, row in enumerate(confirmation_records):
        standardised[index] -= fit["digit_centres"][row["output_digit"]]
    scores = standardised @ fit["direction"]
    summary = conditional_score_summary(scores, confirmation_records, seed=787, bootstrap_samples=200)

    assert summary["output_digit_conditioned_auc"] == 1.0
    assert summary["mean_within_digit_carry_minus_no_carry"] > 0

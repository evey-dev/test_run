import json

import numpy as np

from src.capitals_relation_balanced_localization import (
    candidate_cases,
    choose_split,
    load_intervened_countries,
    paired_score_summary,
    rank_relation_features,
)


def synthetic_records(country_count: int = 12):
    records = []
    for index in range(country_count):
        country = f"country-{index:02d}"
        records.extend(
            [
                {
                    "case_key": country,
                    "country": country,
                    "condition": "capital",
                    "relation_label": 1,
                },
                {
                    "case_key": country,
                    "country": country,
                    "condition": "inverse_country",
                    "relation_label": 0,
                },
            ]
        )
    return records


def test_prior_screen_excludes_only_intervened_countries(tmp_path) -> None:
    result_path = tmp_path / "screen.json"
    result_path.write_text(
        json.dumps(
            {
                "case_selection": {
                    "screened_cases": [{"country": "Rejected"}],
                    "discovery_cases": [{"country": "France"}],
                    "confirmation_cases": [{"country": "Armenia"}],
                }
            }
        ),
        encoding="utf-8",
    )

    countries, sources = load_intervened_countries([result_path])

    assert countries == {"France", "Armenia"}
    assert sources[0]["country_count"] == 2


def test_candidate_variants_are_interleaved_across_countries() -> None:
    rows = [
        {
            "Type": "Country",
            "Location": country,
            "Answer": capital,
            "DistractorAnswer": city,
        }
        for country, capital, city in (
            ("France", "Paris", "Lyon"),
            ("Armenia", "Yerevan", "Gyumri"),
            ("Japan", "Tokyo", "Osaka"),
        )
    ]

    cases = candidate_cases(rows, 7, [], set(), cities_per_country=1)

    assert len({case["country"] for case in cases[:3]}) == 3


def test_split_fallback_is_disjoint_and_baseline_only() -> None:
    eligible = [
        {"country": f"country-{index}", "template": "capital_city"}
        for index in range(40)
    ]

    discovery, confirmation, rule = choose_split(
        eligible,
        desired_per_split=24,
        minimum_per_split=16,
        seed=7,
    )

    assert len(discovery) == len(confirmation) == 20
    assert {row["country"] for row in discovery}.isdisjoint(
        {row["country"] for row in confirmation}
    )
    assert rule["fallback_used"]
    assert rule["fallback_depended_only_on_baseline_eligibility"]


def test_paired_summary_uses_within_country_differences() -> None:
    records = synthetic_records(country_count=8)
    scores = np.asarray(
        [value for _ in range(8) for value in (2.0, 0.5)],
        dtype=np.float32,
    )

    summary = paired_score_summary(scores, records, seed=7)

    assert summary["mean_capital_minus_inverse_score"] == 1.5
    assert summary["paired_relation_accuracy"] == 1.0


def test_all_latent_ranking_prefers_consistent_relation_feature() -> None:
    records = synthetic_records(country_count=12)
    labels = np.asarray([row["relation_label"] for row in records], dtype=np.float32)
    matrix = np.ones((len(records), 48), dtype=np.float32)
    matrix += labels[:, None] * 0.2
    matrix[:, 0] += labels * 2.0

    ranked, statistics, ranking = rank_relation_features(
        matrix,
        records,
        layers=[4, 8],
        minimum_active_fraction=0.1,
        minimum_positive_pair_fraction=0.6,
    )

    assert ranked[0] == 0
    assert int(statistics["latent_dim"].item()) == 24
    assert ranking[0]["key"] == "L4F0"
    assert ranking[0]["positive_discovery_pairs"] == 12

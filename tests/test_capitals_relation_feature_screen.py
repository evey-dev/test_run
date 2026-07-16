from src.capitals_relation_feature_screen import (
    build_panels,
    candidate_cases,
    feature_dict,
    rank_feature_results,
    summarise_rows,
)


def test_candidate_cases_are_country_grouped_and_exclude_graph_entity():
    rows = [
        {
            "Location": "France",
            "Type": "country",
            "Answer": "Paris",
            "DistractorAnswer": "Lyon",
            "sentence": "Fact: The capital of the country containing Lyon is named",
        },
        {
            "Location": "France",
            "Type": "country",
            "Answer": "Paris",
            "DistractorAnswer": "Nice",
            "sentence": "Fact: The capital of the country containing Nice is named",
        },
        {
            "Location": "Jordan",
            "Type": "country",
            "Answer": "Amman",
            "DistractorAnswer": "Zarqa",
            "sentence": "Fact: The capital of the country containing Zarqa is named",
        },
        {
            "Location": "Illinois",
            "Type": "state",
            "Answer": "Springfield",
            "DistractorAnswer": "Chicago",
            "sentence": "Fact: The capital of the state containing Chicago is named",
        },
    ]

    cases = candidate_cases(rows, seed=1, excluded_countries=["Jordan"])

    assert cases
    assert {row["country"] for row in cases} == {"France"}
    assert all(row["capital_prompt_absent_from_sae_corpus"] for row in cases)
    assert all(row["country_prompt_absent_from_sae_corpus"] for row in cases)


def test_candidate_cases_audit_the_actual_sae_corpus():
    rows = [
        {
            "Location": "France",
            "Type": "country",
            "Answer": "Paris",
            "DistractorAnswer": "Lyon",
            "sentence": "Fact: The capital of the country containing Lyon is named",
        }
    ]
    overlapping = {"Fact: The country containing Lyon has a capital named"}

    cases = candidate_cases(
        rows,
        seed=1,
        excluded_countries=[],
        sae_corpus_prompts=overlapping,
    )

    assert any(not row["capital_prompt_absent_from_sae_corpus"] for row in cases)


def test_feature_ranking_prefers_negative_capital_effect_then_specificity():
    rows = [
        {
            "layer": 24,
            "feature": 1,
            "graph_attribution": 2.0,
            "summary": {
                "mean_capital_prompt_delta": -0.2,
                "mean_relation_specific_difference": -0.1,
            },
        },
        {
            "layer": 28,
            "feature": 2,
            "graph_attribution": 1.0,
            "summary": {
                "mean_capital_prompt_delta": -0.2,
                "mean_relation_specific_difference": -0.4,
            },
        },
        {
            "layer": 20,
            "feature": 3,
            "graph_attribution": 3.0,
            "summary": {
                "mean_capital_prompt_delta": 0.5,
                "mean_relation_specific_difference": -1.0,
            },
        },
    ]

    ranked = rank_feature_results(rows)

    assert [(row["layer"], row["feature"]) for row in ranked] == [
        (28, 2),
        (24, 1),
        (20, 3),
    ]


def test_summary_uses_capital_minus_inverse_as_relation_specific_effect():
    rows = [
        {
            "capital_prompt_delta": -0.6,
            "inverse_country_prompt_delta": -0.1,
            "capital_prompt_flipped_to_country": False,
            "capital_prompt_retained_capital": True,
            "inverse_prompt_retained_country": True,
        },
        {
            "capital_prompt_delta": -0.4,
            "inverse_country_prompt_delta": 0.0,
            "capital_prompt_flipped_to_country": True,
            "capital_prompt_retained_capital": False,
            "inverse_prompt_retained_country": True,
        },
    ]

    summary = summarise_rows(rows, seed=7)

    assert summary["mean_capital_prompt_delta"] == -0.5
    assert summary["mean_inverse_country_prompt_delta"] == -0.05
    assert summary["mean_relation_specific_difference"] == -0.45
    assert summary["capital_to_country_flip_fraction"] == 0.5


def test_panels_keep_primary_layer_counts_for_random_controls():
    ranked = [(28, 1), (28, 2), (24, 3), (20, 4)]
    candidates = ranked + [(28, 5), (28, 6), (24, 7), (20, 8)]

    panels = build_panels(
        ranked,
        candidates,
        panel_sizes=[1, 3],
        primary_size=3,
        random_panels=2,
        seed=9,
    )

    primary = next(panel for panel in panels if panel["name"] == "top_3_primary")
    assert feature_dict(primary["features"]) == {24: [3], 28: [1, 2]}
    for panel in panels:
        if panel["kind"] == "layer_count_matched_random_control":
            assert sorted(layer for layer, _ in panel["features"]) == [24, 28, 28]

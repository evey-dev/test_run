import csv
from pathlib import Path

import numpy as np

from src.units_feature_screen import (
    OPERATING_CONTEXTS,
    SYSTEMS,
    baseline_qualification,
    build_confirmation_panels,
    generated_context_cases,
    rank_feature_results,
    select_eligible_cases_by_system,
    summarise_rows,
)


def test_generated_context_cases_are_unique_and_absent_from_sae_corpus() -> None:
    cases = generated_context_cases(seed=2787)
    prompts = {
        prompt
        for case in cases
        for prompt in (case["force_prompt"], case["mass_prompt"], case["energy_prompt"])
    }
    with Path("data/units_data.csv").open("r", encoding="utf-8", newline="") as handle:
        corpus = {row["sentence"] for row in csv.DictReader(handle)}

    assert len(cases) == len(SYSTEMS) * len(OPERATING_CONTEXTS)
    assert len({case["system"] for case in cases}) == len(SYSTEMS) == 64
    assert len(prompts) == 3 * len(cases)
    assert not prompts.intersection(corpus)


def test_mass_control_correctness_is_diagnostic_not_an_eligibility_gate() -> None:
    qualification = baseline_qualification(
        {"top_is_second": True},
        {"top_is_first": True},
        mass_top_id=99,
        mass_expected_ids={10, 11},
    )

    assert qualification["eligible"] is True
    assert qualification["mass_control_top_is_expected"] is False


def test_case_selection_prefers_a_competent_mass_variant() -> None:
    prepared = [
        {
            "system": "system-a",
            "context": "first",
            "eligible": True,
            "mass_control_top_is_expected": False,
        },
        {
            "system": "system-a",
            "context": "second",
            "eligible": True,
            "mass_control_top_is_expected": True,
        },
        {
            "system": "system-b",
            "context": "ineligible",
            "eligible": False,
            "mass_control_top_is_expected": True,
        },
    ]

    selected = select_eligible_cases_by_system(prepared)

    assert selected["system-a"]["context"] == "second"
    assert "system-b" not in selected


def test_units_summary_and_ranking_use_force_minus_mass_difference() -> None:
    selective_rows = [
        {
            "force_source_delta": 0.5,
            "mass_source_delta": 0.1,
            "force_top_prediction_transfer": False,
            "mass_top_prediction_transfer": False,
        }
        for _ in range(4)
    ]
    generic_rows = [
        {
            "force_source_delta": 0.5,
            "mass_source_delta": 0.6,
            "force_top_prediction_transfer": False,
            "mass_top_prediction_transfer": False,
        }
        for _ in range(4)
    ]
    selective = summarise_rows(selective_rows, seed=787)
    generic = summarise_rows(generic_rows, seed=787)
    ranked = rank_feature_results(
        [
            {"layer": 8, "feature": 2, "graph_attribution": 0.2, "summary": generic},
            {"layer": 4, "feature": 1, "graph_attribution": 0.1, "summary": selective},
        ]
    )

    assert np.isclose(selective["mean_force_minus_mass_difference"], 0.4)
    assert ranked[0]["feature"] == 1


def test_units_confirmation_panels_have_fixed_primary_and_random_controls() -> None:
    ranked = [(4, 1), (4, 2), (8, 3), (8, 4), (12, 5)]
    panels = build_confirmation_panels(
        ranked,
        ranked,
        panel_sizes=[1, 3],
        primary_panel_size=3,
        random_panels=2,
        seed=787,
    )
    by_name = {panel["name"]: panel for panel in panels}

    assert by_name["top_3_primary"]["features"] == ranked[:3]
    assert len(by_name["random_matched_01"]["features"]) == 3
    assert sorted(layer for layer, _ in by_name["random_matched_01"]["features"]) == [4, 4, 8]

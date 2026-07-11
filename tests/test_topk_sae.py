import numpy as np
import torch

from src.heldout_validation import summarise_math_specificity
from src.math_carry_feature_screen import (
    build_confirmation_panels,
    feature_dict,
    rank_feature_results,
    summarise_panel,
)
from src.train import SparseAutoencoder


def test_relu_encoding_matches_legacy_formula() -> None:
    torch.manual_seed(787)
    model = SparseAutoencoder(16, 32)
    x = torch.randn(4, 16)

    expected = torch.relu(model.encoder(x - model.decoder_bias))

    assert torch.equal(model.encode(x), expected)


def test_topk_bounds_l0_and_decoder_projection() -> None:
    torch.manual_seed(787)
    model = SparseAutoencoder(16, 32, activation_type="topk", top_k=5)
    x = torch.randn(8, 16)

    z = model.encode(x)
    model.normalize_decoder_columns()

    assert torch.all((z > 1e-6).sum(dim=-1) <= 5)
    assert torch.allclose(
        model.decoder.weight.norm(dim=0),
        torch.ones(32),
        atol=1e-6,
    )


def test_specificity_summary_uses_target_minus_control() -> None:
    rows = []
    for target_delta, control_delta in [(-0.50, 0.00), (-0.25, 0.10), (-0.75, -0.10)]:
        rows.append(
            {
                "eligible": True,
                "conditions": {
                    "clean": {"gap": 5.0},
                    "sparse_inhibition": {"gap": 5.0 + target_delta},
                },
                "specificity_control": {"gap_delta": control_delta},
            }
        )

    summary = summarise_math_specificity(rows, seed=787)

    assert np.isclose(summary["mean_target_delta"], -0.50)
    assert np.isclose(summary["mean_no_carry_control_delta"], 0.00)
    assert np.isclose(summary["mean_paired_difference"], -0.50)
    assert summary["fraction_target_more_negative_than_control"] == 1.0


def test_feature_screen_summary_and_discovery_ranking() -> None:
    selective_rows = [
        {
            "target_delta": -0.5,
            "control_delta": -0.1,
            "target_top_transferred": False,
            "control_top_transferred": False,
            "target_activation": 1.0,
            "control_activation": 0.2,
        }
        for _ in range(4)
    ]
    generic_rows = [
        {
            "target_delta": -0.5,
            "control_delta": -0.6,
            "target_top_transferred": False,
            "control_top_transferred": False,
            "target_activation": 1.0,
            "control_activation": 1.0,
        }
        for _ in range(4)
    ]
    selective = summarise_panel(selective_rows, seed=787)
    generic = summarise_panel(generic_rows, seed=787)
    ranked = rank_feature_results(
        [
            {"layer": 8, "feature": 2, "graph_attribution": 0.2, "summary": generic},
            {"layer": 4, "feature": 1, "graph_attribution": 0.1, "summary": selective},
        ]
    )

    assert np.isclose(selective["mean_paired_difference"], -0.4)
    assert ranked[0]["feature"] == 1


def test_confirmation_panels_are_prefixes_with_matched_random_size() -> None:
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

    assert feature_dict(by_name["top_3_primary"]["features"]) == {4: [1, 2], 8: [3]}
    assert len(by_name["random_matched_01"]["features"]) == 3
    assert sorted(layer for layer, _ in by_name["random_matched_01"]["features"]) == [4, 4, 8]

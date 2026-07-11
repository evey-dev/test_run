import numpy as np
import torch

from src.heldout_validation import summarise_math_specificity
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

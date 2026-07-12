from pathlib import Path

import pytest

from src.plot_attribution_graph import (
    backward_connected_subset,
    layer_sort_key,
    panel_feature_ids,
    retained_edges,
)
from src.prompts.prompt_utils import (
    format_prompt,
    get_expected_answers,
    get_expected_token,
    load_prompts,
    load_prompts_from_csv,
)


def test_prompt_csv_accepts_supported_column_names(tmp_path: Path) -> None:
    csv_path = tmp_path / "prompts.csv"
    csv_path.write_text(
        "sentence,Expected\nQuestion: 2 + 2?,4\n",
        encoding="utf-8",
    )

    rows = load_prompts_from_csv(csv_path)

    assert rows[0]["sentence"] == "Question: 2 + 2?"
    assert rows[0]["Expected"] == 4


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("Answer\n4\n", "sentence"),
        ("sentence\nQuestion?\n", "answer column"),
    ],
)
def test_prompt_csv_rejects_missing_required_columns(
    tmp_path: Path, contents: str, message: str
) -> None:
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text(contents, encoding="utf-8")

    with pytest.raises(KeyError, match=message):
        load_prompts_from_csv(csv_path)


def test_prompt_formatting_strips_whitespace_and_rejects_empty_text() -> None:
    assert format_prompt({"sentence": "  Question?  "}) == "Question?"
    with pytest.raises(ValueError, match="missing a sentence"):
        format_prompt({"sentence": "   "})


def test_expected_answers_parse_serialised_lists_and_choose_first_token() -> None:
    prompt = {"Answer": "['newtons', 'N']"}

    assert get_expected_answers(prompt) == ["newtons", "N"]
    assert get_expected_token(prompt) == "newtons"


def test_expected_answers_support_alternate_keys_and_reject_missing_value() -> None:
    assert get_expected_answers({"Capital": "Paris"}) == ["Paris"]
    with pytest.raises(KeyError, match="expected token"):
        get_expected_answers({"sentence": "Question?"})


def test_prompt_loader_rejects_unknown_behaviour() -> None:
    with pytest.raises(ValueError, match="Unsupported behaviour"):
        load_prompts("unknown-domain")


def synthetic_graph():
    nodes = [
        {"id": "input_a", "layer": "input", "attribution": 0.2},
        {"id": "input_b", "layer": "input", "attribution": 0.1},
        {"id": "layer_4_feature_1", "layer": "layer_4", "attribution": 0.5},
        {"id": "layer_4_feature_2", "layer": "layer_4", "attribution": 0.4},
        {"id": "target_logit", "layer": "logits", "attribution": 1.0},
    ]
    edges = [
        {"source": "input_a", "target": "layer_4_feature_1", "weight": 0.9},
        {"source": "input_b", "target": "layer_4_feature_2", "weight": 0.2},
        {"source": "layer_4_feature_1", "target": "target_logit", "weight": 0.8},
        {"source": "layer_4_feature_2", "target": "target_logit", "weight": 0.1},
    ]
    return nodes, edges


def test_layer_sort_key_orders_inputs_features_and_logits() -> None:
    layers = ["logits", "layer_12", "input", "layer_4"]

    assert sorted(layers, key=layer_sort_key) == [
        "input",
        "layer_4",
        "layer_12",
        "logits",
    ]


def test_backward_subset_follows_strongest_connected_path() -> None:
    nodes, edges = synthetic_graph()

    layers, selected = backward_connected_subset(nodes, edges, nodes_per_layer=1)

    assert layers == ["input", "layer_4", "logits"]
    assert selected == {"input_a", "layer_4_feature_1", "target_logit"}


def test_backward_subset_forces_required_panel_nodes_into_view() -> None:
    nodes, edges = synthetic_graph()

    _, selected = backward_connected_subset(
        nodes,
        edges,
        nodes_per_layer=1,
        required_ids={"layer_4_feature_2"},
    )

    assert "layer_4_feature_2" in selected
    assert "layer_4_feature_1" not in selected


def test_backward_subset_rejects_unknown_required_node() -> None:
    nodes, edges = synthetic_graph()

    with pytest.raises(ValueError, match="absent"):
        backward_connected_subset(
            nodes,
            edges,
            nodes_per_layer=1,
            required_ids={"missing_feature"},
        )


def test_retained_edges_keep_largest_magnitude_per_transition() -> None:
    nodes, edges = synthetic_graph()
    node_by_id = {node["id"]: node for node in nodes}

    selected = {node["id"] for node in nodes}
    retained = retained_edges(edges, selected, node_by_id, edges_per_transition=1)

    assert {(edge["source"], edge["target"]) for edge in retained} == {
        ("input_a", "layer_4_feature_1"),
        ("layer_4_feature_1", "target_logit"),
    }


def test_panel_feature_ids_extract_named_panel_and_reject_unknown_name() -> None:
    screen = {
        "confirmation": {
            "panels": [
                {
                    "name": "top_2_primary",
                    "features": [
                        {"layer": 4, "feature": 11},
                        {"layer": 28, "feature": 22},
                    ],
                }
            ]
        }
    }

    assert panel_feature_ids(screen, "top_2_primary") == {
        "layer_4_feature_11",
        "layer_28_feature_22",
    }
    with pytest.raises(ValueError, match="absent"):
        panel_feature_ids(screen, "missing")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config_utils import ensure_output_dir, load_yaml_config
from src.data_utils import (
    find_activation_data_dir,
    load_activation_splits,
    load_activation_tensor,
    resolve_path,
    save_activation_metadata,
    verify_activation_file,
)


def test_yaml_loader_handles_mapping_and_empty_file(tmp_path: Path) -> None:
    populated = tmp_path / "config.yaml"
    populated.write_text("seed: 787\nlayers: [4, 8]\n", encoding="utf-8")
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")

    assert load_yaml_config(populated) == {"seed": 787, "layers": [4, 8]}
    assert load_yaml_config(empty) == {}


def test_ensure_output_dir_is_nested_and_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "results"

    assert ensure_output_dir(output) == output
    assert ensure_output_dir(output) == output
    assert output.is_dir()


def test_resolve_path_respects_base_directory_and_absolute_paths(tmp_path: Path) -> None:
    absolute = tmp_path / "absolute.json"

    assert resolve_path("results/value.json", tmp_path) == (
        tmp_path / "results" / "value.json"
    ).resolve()
    assert resolve_path(absolute, tmp_path) == absolute


def test_activation_file_metadata_reports_shape_dtype_and_moments(tmp_path: Path) -> None:
    path = tmp_path / "activations_layer4.npy"
    values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np.save(path, values)

    metadata = verify_activation_file(path)

    assert metadata["shape"] == (2, 2)
    assert metadata["dtype"] == "float32"
    assert np.isclose(metadata["mean"], values.mean())
    assert np.isclose(metadata["std"], values.std())


def test_activation_directory_prefers_explicit_valid_bundle(tmp_path: Path) -> None:
    np.save(tmp_path / "activations_layer8.npy", np.zeros((2, 3), dtype=np.float32))

    assert find_activation_data_dir(tmp_path) == tmp_path.resolve()


def test_activation_tensor_loads_requested_layer_and_reports_missing_file(
    tmp_path: Path,
) -> None:
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.save(tmp_path / "activations_layer12.npy", expected)

    assert np.array_equal(load_activation_tensor(12, tmp_path), expected)
    with pytest.raises(FileNotFoundError, match="activations_layer16.npy"):
        load_activation_tensor(16, tmp_path)


def test_activation_splits_normalise_layer_keys_and_integer_indices(
    tmp_path: Path,
) -> None:
    raw = {
        "4": {"train": [0, 2, 4], "val": [1, 3]},
        8: {"train": np.asarray([1, 3]), "val": np.asarray([0, 2])},
    }
    np.save(tmp_path / "train_val_indices_per_layer.npy", raw)

    splits = load_activation_splits(tmp_path)

    assert set(splits) == {4, 8}
    assert np.array_equal(splits[4]["train"], np.asarray([0, 2, 4]))
    assert np.issubdtype(splits[8]["val"].dtype, np.integer)


def test_activation_metadata_round_trips_to_csv(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    rows = [
        {"prompt_id": "units-0", "layer": 4},
        {"prompt_id": "units-1", "layer": 8},
    ]

    save_activation_metadata(path, rows)
    recovered = pd.read_csv(path).to_dict(orient="records")

    assert recovered == rows

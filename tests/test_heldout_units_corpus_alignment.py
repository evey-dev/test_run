import csv
from pathlib import Path

import numpy as np
import yaml

from src.heldout_validation import generated_unit_cases


def test_units_cases_use_indices_from_the_matching_corpus(tmp_path: Path) -> None:
    rows = [
        {
            "Quantity": "force",
            "ContextObject": f"apparatus-{index}",
            "sentence": (
                f'Fact: The official SI unit for the force of apparatus-{index} is named "'
            ),
        }
        for index in range(25)
    ]
    corpus_path = tmp_path / "units.csv"
    with corpus_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    splits = {4: {"train": np.asarray([], dtype=int), "val": np.arange(len(rows))}}
    np.save(tmp_path / "train_val_indices_per_layer.npy", splits, allow_pickle=True)
    config_path = tmp_path / "sae.yaml"
    config_path.write_text(yaml.safe_dump({"data_dir": str(tmp_path)}), encoding="utf-8")

    cases = generated_unit_cases(20, corpus_path, config_path, seed=787)

    assert len(cases) == 20
    assert len({case["context_object"] for case in cases}) == 20
    assert all(case["source_sae_split"] == "validation" for case in cases)

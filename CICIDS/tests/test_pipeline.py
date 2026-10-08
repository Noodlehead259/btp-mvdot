from pathlib import Path

import numpy as np

from src.create_alignment import build_alignment_map
from src.evaluate import evaluate
from src.preprocess import load_config, write_view_csv


ROOT = Path(__file__).resolve().parents[1]


def test_config_has_four_disjoint_label_free_views():
    config = load_config(ROOT / "configs" / "views.json")
    flattened = [feature for features in config["views"].values() for feature in features]
    assert len(config["views"]) == 4
    assert len(flattened) == 60
    assert len(flattened) == len(set(flattened))
    assert config["label_column"] not in flattened
    assert [len(features) for features in config["views"].values()] == [7, 16, 14, 23]


def test_alignment_map_is_reproducible_and_has_exact_aligned_count():
    row_count = 100
    for fraction in (1.0, 0.75, 0.5, 0.25, 0.0):
        first = build_alignment_map(row_count, fraction, np.random.default_rng(15))
        second = build_alignment_map(row_count, fraction, np.random.default_rng(15))
        row_indices, aligned = first
        assert np.array_equal(row_indices, second[0])
        assert np.array_equal(aligned, second[1])
        assert int(aligned.sum()) == int(np.floor(row_count * fraction + 0.5))
        assert np.array_equal(np.flatnonzero(row_indices == np.arange(row_count)), np.flatnonzero(aligned))
        assert sorted(row_indices.tolist()) == list(range(row_count))


def test_accuracy_is_invariant_to_cluster_id_permutation():
    labels = np.array(["a", "a", "b", "b", "c", "c"])
    predictions = np.array([22, 22, 5, 5, 11, 11])
    metrics = evaluate(labels, predictions)
    assert metrics["ACC"] == 1.0
    assert metrics["NMI"] == 1.0
    assert metrics["ARI"] == 1.0


def test_view_csv_includes_canonical_row_indices_and_feature_headers(tmp_path):
    values = np.array([[0.25, -0.5], [1.0, 2.0], [-1.5, 0.75]])
    output = tmp_path / "view.csv.gz"
    write_view_csv(output, values, ["bytes", "duration"], chunk_size=2)

    import pandas as pd

    frame = pd.read_csv(output)
    assert frame.columns.tolist() == ["row_index", "bytes", "duration"]
    assert frame["row_index"].tolist() == [0, 1, 2]
    np.testing.assert_allclose(frame[["bytes", "duration"]].to_numpy(), values)
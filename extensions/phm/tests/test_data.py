from pathlib import Path

import numpy as np
from phm_mcp.data import (
    build_test_windows,
    build_windows,
    fit_preprocessor,
    load_cmapss_file,
    split_engine_ids,
    training_labels,
    transform_features,
)


def rows(engine_count: int = 4, cycles: int = 5) -> np.ndarray:
    output = []
    for unit_id in range(1, engine_count + 1):
        for cycle in range(1, cycles + 1):
            features = np.arange(24, dtype=np.float32) + cycle + unit_id
            features[0] = 1.0
            output.append([unit_id, cycle, *features])
    return np.asarray(output, dtype=np.float32)


def test_training_labels_are_capped_and_end_at_zero() -> None:
    labels = training_labels(rows(engine_count=2, cycles=5), cap=3)
    assert labels.tolist() == [3, 3, 2, 1, 0, 3, 3, 2, 1, 0]


def test_engine_split_has_no_leakage() -> None:
    train, validation = split_engine_ids(rows(), validation_fraction=0.25, seed=7)
    assert set(train).isdisjoint(validation)
    assert sorted([*train, *validation]) == [1, 2, 3, 4]


def test_preprocessor_is_fit_only_on_training_engines() -> None:
    values = rows()
    train_ids = np.asarray([1, 2, 3])
    selected, mean, scale = fit_preprocessor(values, train_ids)
    transformed = transform_features(values, selected, mean, scale)
    train_mask = np.isin(values[:, 0].astype(int), train_ids)
    np.testing.assert_allclose(transformed[train_mask].mean(axis=0), 0, atol=1e-6)
    assert 0 not in selected


def test_window_and_test_label_alignment() -> None:
    values = rows(engine_count=2, cycles=5)
    selected, mean, scale = fit_preprocessor(values, np.asarray([1, 2]))
    features = transform_features(values, selected, mean, scale)
    labels = training_labels(values)
    x, y = build_windows(values, features, labels, np.asarray([1]), window_size=3)
    assert x.shape == (3, 3, len(selected))
    assert y.tolist() == [2, 1, 0]
    test_x, test_y, ids, last_cycles = build_test_windows(
        values,
        features,
        np.asarray([10, 20], dtype=np.float32),
        window_size=3,
    )
    assert test_x.shape[0] == 2
    assert test_y.tolist() == [10, 20]
    assert ids.tolist() == [1, 2]
    assert last_cycles.tolist() == [5, 5]


def test_loader_rejects_wrong_column_count(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    np.savetxt(path, np.ones((2, 25)))
    try:
        load_cmapss_file(path)
    except ValueError as error:
        assert "26 columns" in str(error)
    else:
        raise AssertionError("invalid data was accepted")

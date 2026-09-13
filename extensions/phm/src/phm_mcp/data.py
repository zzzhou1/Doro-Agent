from __future__ import annotations

import hashlib
import json
import random
import shutil
import urllib.error
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import (
    FD001_FILES,
    FD001_MIRROR_URL,
    FD001_SHA256,
    NASA_CMAPSS_URL,
    RUL_CAP,
    WINDOW_SIZE,
)


@dataclass(frozen=True)
class PreparedSummary:
    output_path: Path
    train_engines: int
    validation_engines: int
    test_engines: int
    feature_count: int
    train_windows: int
    validation_windows: int


def column_names() -> list[str]:
    return [
        "unit_id",
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3",
        *[f"sensor_{i}" for i in range(1, 22)],
    ]


def download_fd001(
    data_dir: Path,
    url: str = NASA_CMAPSS_URL,
    mirror_url: str = FD001_MIRROR_URL,
) -> list[Path]:
    """Download FD001 from NASA, with a pinned checksum-verified fallback."""
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = [raw_dir / name for name in FD001_FILES]
    if _verify_fd001(raw_dir):
        return existing

    archive = data_dir / "cmapss.zip.part"
    try:
        with urllib.request.urlopen(url, timeout=30) as response, archive.open("wb") as target:
            shutil.copyfileobj(response, target)
        _extract_fd001(archive, raw_dir)
        if not _verify_fd001(raw_dir):
            raise ValueError("Official FD001 files failed checksum verification")
    except (
        OSError,
        TimeoutError,
        urllib.error.URLError,
        zipfile.BadZipFile,
        ValueError,
    ) as error:
        warnings.warn(
            f"NASA archive unavailable ({error}); using the pinned FD001 mirror.",
            RuntimeWarning,
            stacklevel=2,
        )
        _download_mirror(raw_dir, mirror_url)
    finally:
        archive.unlink(missing_ok=True)

    if not _verify_fd001(raw_dir):
        raise ValueError("FD001 files failed checksum verification")
    return existing


def load_cmapss_file(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"C-MAPSS file not found: {path}")
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 26:
        raise ValueError(f"Expected 26 columns in {path.name}, got shape {data.shape}")
    return data


def load_rul_file(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"RUL file not found: {path}")
    values = np.loadtxt(path, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError(f"RUL file is empty: {path}")
    return values


def training_labels(rows: np.ndarray, cap: float = RUL_CAP) -> np.ndarray:
    labels = np.empty(rows.shape[0], dtype=np.float32)
    unit_ids = rows[:, 0].astype(np.int64)
    cycles = rows[:, 1]
    for unit_id in np.unique(unit_ids):
        mask = unit_ids == unit_id
        labels[mask] = np.minimum(cycles[mask].max() - cycles[mask], cap)
    return labels


def split_engine_ids(
    rows: np.ndarray, validation_fraction: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    ids = sorted(int(value) for value in np.unique(rows[:, 0]))
    if len(ids) < 2:
        raise ValueError("At least two engines are required for train/validation splitting")
    rng = random.Random(seed)
    rng.shuffle(ids)
    validation_count = min(len(ids) - 1, max(1, round(len(ids) * validation_fraction)))
    validation_ids = np.asarray(sorted(ids[:validation_count]), dtype=np.int64)
    train_ids = np.asarray(sorted(ids[validation_count:]), dtype=np.int64)
    return train_ids, validation_ids


def fit_preprocessor(
    rows: np.ndarray, train_ids: np.ndarray, variance_epsilon: float = 1e-8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_rows = rows[np.isin(rows[:, 0].astype(np.int64), train_ids), 2:]
    if feature_rows.size == 0:
        raise ValueError("No rows matched the training engine IDs")
    std = feature_rows.std(axis=0)
    selected = np.flatnonzero(std > variance_epsilon).astype(np.int64)
    if selected.size == 0:
        raise ValueError("No non-constant C-MAPSS features remain")
    selected_rows = feature_rows[:, selected]
    mean = selected_rows.mean(axis=0).astype(np.float32)
    scale = selected_rows.std(axis=0).astype(np.float32)
    scale[scale < variance_epsilon] = 1.0
    return selected, mean, scale


def transform_features(
    rows: np.ndarray, selected: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return ((rows[:, 2:][:, selected] - mean) / scale).astype(np.float32)


def build_windows(
    rows: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    engine_ids: np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    targets: list[float] = []
    row_units = rows[:, 0].astype(np.int64)
    for unit_id in engine_ids:
        indices = np.flatnonzero(row_units == int(unit_id))
        indices = indices[np.argsort(rows[indices, 1])]
        for end in range(window_size - 1, len(indices)):
            choice = indices[end - window_size + 1 : end + 1]
            windows.append(features[choice])
            targets.append(float(labels[choice[-1]]))
    if not windows:
        raise ValueError(f"No {window_size}-cycle windows could be built")
    return np.stack(windows).astype(np.float32), np.asarray(targets, dtype=np.float32)


def build_test_windows(
    rows: np.ndarray,
    features: np.ndarray,
    true_rul: np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_units = rows[:, 0].astype(np.int64)
    ids = np.unique(row_units).astype(np.int64)
    if len(ids) != len(true_rul):
        raise ValueError(
            f"Test engine count ({len(ids)}) does not match RUL labels ({len(true_rul)})"
        )
    windows: list[np.ndarray] = []
    last_cycles: list[int] = []
    for unit_id in ids:
        indices = np.flatnonzero(row_units == unit_id)
        indices = indices[np.argsort(rows[indices, 1])]
        if len(indices) < window_size:
            raise ValueError(f"Engine {unit_id} has fewer than {window_size} cycles")
        choice = indices[-window_size:]
        windows.append(features[choice])
        last_cycles.append(int(rows[choice[-1], 1]))
    return (
        np.stack(windows).astype(np.float32),
        true_rul.astype(np.float32),
        ids,
        np.asarray(last_cycles, dtype=np.int64),
    )


def _verify_fd001(raw_dir: Path) -> bool:
    return all(
        (path := raw_dir / name).is_file() and _sha256(path) == expected
        for name, expected in FD001_SHA256.items()
    )


def _download_mirror(raw_dir: Path, mirror_url: str) -> None:
    for name, expected in FD001_SHA256.items():
        destination = raw_dir / name
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with (
                urllib.request.urlopen(f"{mirror_url}/{name}", timeout=120) as response,
                temporary.open("wb") as target,
            ):
                shutil.copyfileobj(response, target)
            if _sha256(temporary) != expected:
                raise ValueError(f"Checksum mismatch for mirrored file: {name}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def _extract_fd001(archive: Path, raw_dir: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        members = {Path(name).name: name for name in bundle.namelist()}
        missing = [name for name in FD001_FILES if name not in members]
        if missing:
            raise ValueError(f"NASA archive is missing: {', '.join(missing)}")
        for name in FD001_FILES:
            temporary = (raw_dir / name).with_suffix(".txt.part")
            try:
                with bundle.open(members[name]) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
                temporary.replace(raw_dir / name)
            finally:
                temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_fd001(
    data_dir: Path,
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
    window_size: int = WINDOW_SIZE,
) -> PreparedSummary:
    raw_dir = data_dir / "raw"
    train_path, test_path, rul_path = [raw_dir / name for name in FD001_FILES]
    train_rows = load_cmapss_file(train_path)
    test_rows = load_cmapss_file(test_path)
    test_rul = load_rul_file(rul_path)

    train_ids, validation_ids = split_engine_ids(train_rows, validation_fraction, seed)
    selected, mean, scale = fit_preprocessor(train_rows, train_ids)
    train_features = transform_features(train_rows, selected, mean, scale)
    test_features = transform_features(test_rows, selected, mean, scale)
    labels = training_labels(train_rows)
    train_x, train_y = build_windows(train_rows, train_features, labels, train_ids, window_size)
    validation_x, validation_y = build_windows(
        train_rows, train_features, labels, validation_ids, window_size
    )
    test_x, test_y, test_ids, test_cycles = build_test_windows(
        test_rows, test_features, test_rul, window_size
    )

    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "fd001.npz"
    np.savez_compressed(
        output_path,
        train_x=train_x,
        train_y=train_y,
        validation_x=validation_x,
        validation_y=validation_y,
        test_x=test_x,
        test_y=test_y,
        test_unit_ids=test_ids,
        test_last_cycles=test_cycles,
        train_unit_ids=train_ids,
        validation_unit_ids=validation_ids,
        selected_feature_indices=selected,
        feature_mean=mean,
        feature_scale=scale,
        window_size=np.asarray(window_size),
    )
    manifest = {
        "dataset": "FD001",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "window_size": window_size,
        "rul_cap": RUL_CAP,
        "train_unit_ids": train_ids.tolist(),
        "validation_unit_ids": validation_ids.tolist(),
        "selected_features": [column_names()[index + 2] for index in selected],
        "raw_sha256": {path.name: _sha256(path) for path in (train_path, test_path, rul_path)},
    }
    (processed_dir / "fd001_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return PreparedSummary(
        output_path=output_path,
        train_engines=len(train_ids),
        validation_engines=len(validation_ids),
        test_engines=len(test_ids),
        feature_count=len(selected),
        train_windows=len(train_x),
        validation_windows=len(validation_x),
    )

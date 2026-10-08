"""Build independently normalized, label-free CIC-IDS2017 feature views."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    views = config["views"]
    flattened = [feature for features in views.values() for feature in features]
    if len(flattened) != len(set(flattened)):
        raise ValueError("A feature may belong to only one view.")
    if config["label_column"] in flattened:
        raise ValueError("The label column cannot be included in a feature view.")
    return config


def _clean_columns(columns: pd.Index) -> list[str]:
    cleaned = [str(column).strip() for column in columns]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError("Column names collide after trimming whitespace.")
    return cleaned


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_view_csv(
    path: Path,
    values: np.ndarray,
    feature_names: list[str],
    chunk_size: int = 50_000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        for start in range(0, len(values), chunk_size):
            stop = min(start + chunk_size, len(values))
            frame = pd.DataFrame(values[start:stop], columns=feature_names)
            frame.insert(0, "row_index", np.arange(start, stop))
            frame.to_csv(handle, index=False, header=start == 0)


def _prepare_views(
    chunks_by_view: dict[str, list[np.ndarray]],
    feature_lists: dict[str, list[str]],
) -> tuple[dict[str, np.ndarray], dict[str, list[str]], dict[str, dict[str, float]]]:
    arrays: dict[str, np.ndarray] = {}
    retained_features: dict[str, list[str]] = {}
    imputation_values: dict[str, dict[str, float]] = {}

    for view_name, chunks in chunks_by_view.items():
        raw = np.concatenate(chunks, axis=0)
        keep = ~np.isnan(raw).all(axis=0)
        feature_names = [name for name, should_keep in zip(feature_lists[view_name], keep) if should_keep]
        raw = raw[:, keep]
        if not feature_names:
            raise ValueError(f"No valid numerical features remain in {view_name}.")

        medians = np.nanmedian(raw, axis=0)
        invalid_medians = ~np.isfinite(medians)
        if invalid_medians.any():
            raise ValueError(f"Could not calculate finite medians for {view_name}.")
        missing_rows, missing_cols = np.where(np.isnan(raw))
        raw[missing_rows, missing_cols] = medians[missing_cols]

        scaler = StandardScaler()
        arrays[view_name] = scaler.fit_transform(raw).astype(np.float32, copy=False)
        retained_features[view_name] = feature_names
        imputation_values[view_name] = {
            name: float(value) for name, value in zip(feature_names, medians)
        }

    return arrays, retained_features, imputation_values


def preprocess(
    source_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    chunk_size: int = 50_000,
) -> dict[str, Any]:
    files = sorted(source_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")

    feature_lists: dict[str, list[str]] = config["views"]
    expected_columns = {config["label_column"]}
    expected_columns.update(feature for values in feature_lists.values() for feature in values)
    row_hashes: set[int] = set()
    view_chunks: dict[str, list[np.ndarray]] = {name: [] for name in feature_lists}
    labels: list[str] = []
    source_files: list[str] = []
    duplicate_rows = 0
    raw_rows = 0

    for path in files:
        print(f"Reading {path.name}", flush=True)
        header = _clean_columns(pd.read_csv(path, nrows=0).columns)
        missing_columns = expected_columns.difference(header)
        if missing_columns:
            raise ValueError(f"{path.name} is missing required columns: {sorted(missing_columns)}")

        for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
            chunk.columns = _clean_columns(chunk.columns)
            raw_rows += len(chunk)
            hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy(dtype="uint64")
            unique_mask = np.ones(len(chunk), dtype=bool)
            for index, value in enumerate(hashes):
                key = int(value)
                if key in row_hashes:
                    unique_mask[index] = False
                    duplicate_rows += 1
                else:
                    row_hashes.add(key)
            if not unique_mask.any():
                continue

            retained = chunk.loc[unique_mask]
            labels.extend(retained[config["label_column"]].astype("string").fillna("").tolist())
            source_files.extend([path.name] * len(retained))
            for view_name, features in feature_lists.items():
                numeric = retained[features].apply(pd.to_numeric, errors="coerce")
                values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
                values[~np.isfinite(values)] = np.nan
                view_chunks[view_name].append(values)

    arrays, retained_features, imputation_values = _prepare_views(view_chunks, feature_lists)
    if any(len(array) != len(labels) for array in arrays.values()):
        raise RuntimeError("View and label row counts diverged during preprocessing.")

    views_dir = output_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    for view_name, array in arrays.items():
        np.save(views_dir / f"{view_name}.npy", array, allow_pickle=False)
        write_view_csv(
            views_dir / f"{view_name}.csv.gz",
            array,
            retained_features[view_name],
            chunk_size,
        )
    labels_frame = pd.DataFrame({"row_index": np.arange(len(labels)), "label": labels, "source_file": source_files})
    labels_frame.to_csv(output_dir / "labels.csv.gz", index=False, compression="gzip")

    metadata = {
        "source_files": [
            {"name": path.name, "size_bytes": path.stat().st_size} for path in files
        ],
        "raw_row_count": raw_rows,
        "retained_row_count": len(labels),
        "exact_duplicate_rows_removed": duplicate_rows,
        "label_column": config["label_column"],
        "label_used_for_view_construction": False,
        "feature_count": sum(len(values) for values in retained_features.values()),
        "features_per_view": {name: len(values) for name, values in retained_features.items()},
        "views": {
            name: {
                "features": retained_features[name],
                "shape": list(arrays[name].shape),
                "dtype": str(arrays[name].dtype),
                "normalization": "StandardScaler fitted independently for this view",
                "median_imputation_values": imputation_values[name],
                "file": f"views/{name}.npy",
                "csv_file": f"views/{name}.csv.gz",
            }
            for name in arrays
        },
        "labels_file": "labels.csv.gz",
        "decisions": config["decisions"],
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    checksums = {}
    output_files = [
        *(views_dir / f"{name}.npy" for name in arrays),
        *(views_dir / f"{name}.csv.gz" for name in arrays),
        output_dir / "labels.csv.gz",
    ]
    for path in output_files:
        checksums[str(path.relative_to(output_dir))] = _sha256(path)
    metadata["sha256"] = checksums
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(arrays)} view arrays, CSVs, and {len(labels):,} labels to {output_dir}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/views.json"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()
    config = load_config(args.config)
    source_dir = args.source_dir or Path(config["source_dir"])
    output_dir = args.output_dir or Path(config["output_dir"])
    preprocess(source_dir, output_dir, config, args.chunk_size)


if __name__ == "__main__":
    main()
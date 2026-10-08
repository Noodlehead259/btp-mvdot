"""Generate a full, streaming inspection report for CIC-IDS2017 CSV files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _clean_columns(columns: pd.Index) -> list[str]:
    cleaned = [str(column).strip() for column in columns]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError("Column names collide after trimming whitespace.")
    return cleaned


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _scan_file(path: Path, chunk_size: int) -> dict[str, Any]:
    raw_header = pd.read_csv(path, nrows=0).columns
    columns = _clean_columns(raw_header)
    types: dict[str, set[str]] = {column: set() for column in columns}
    missing: Counter[str] = Counter()
    infinite: Counter[str] = Counter()
    candidate_values: dict[str, set[Any]] = {column: set() for column in columns}
    label_counts: Counter[str] = Counter()
    duplicate_hashes: set[int] = set()
    duplicate_rows = 0
    rows = 0
    near_candidates: set[str] = set()

    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        chunk.columns = _clean_columns(chunk.columns)
        rows += len(chunk)
        hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy(dtype="uint64")
        for value in hashes:
            key = int(value)
            if key in duplicate_hashes:
                duplicate_rows += 1
            else:
                duplicate_hashes.add(key)

        for column in columns:
            series = chunk[column]
            types[column].add(str(series.dtype))
            missing[column] += int(series.isna().sum())
            numeric = pd.to_numeric(series, errors="coerce")
            if pd.api.types.is_numeric_dtype(series) or numeric.notna().any():
                values = numeric.to_numpy(dtype="float64", na_value=np.nan)
                infinite[column] += int(np.isinf(values).sum())
                current = candidate_values[column]
                if len(current) <= 1:
                    for value in numeric.dropna().unique():
                        current.add(value)
                        if len(current) == 2:
                            break
            if len(series):
                most_common = series.value_counts(dropna=False).iloc[0]
                if most_common / len(series) >= 0.99:
                    near_candidates.add(column)

        if "Label" in chunk.columns:
            label_counts.update(
                chunk["Label"].astype("string").fillna("<NA>").value_counts().to_dict()
            )

    near_counts: dict[str, Counter[Any]] = {column: Counter() for column in near_candidates}
    if near_candidates:
        for chunk in pd.read_csv(
            path,
            chunksize=chunk_size,
            low_memory=False,
            usecols=lambda column: column.strip() in near_candidates,
        ):
            chunk.columns = _clean_columns(chunk.columns)
            for column in near_candidates:
                near_counts[column].update(chunk[column].value_counts(dropna=False).to_dict())

    constants = {
        column: list(values) if values else []
        for column, values in candidate_values.items()
        if len(values) <= 1
    }
    near_constants = {}
    for column, counts in near_counts.items():
        if counts:
            value, count = counts.most_common(1)[0]
            fraction = count / rows
            if fraction >= 0.99:
                near_constants[column] = {
                    "value": str(value),
                    "count": int(count),
                    "fraction": fraction,
                }

    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "rows": rows,
        "column_count": len(columns),
        "columns": columns,
        "data_types": {column: sorted(values) for column, values in types.items()},
        "missing_values": {column: int(missing[column]) for column in columns},
        "infinite_values": {column: int(infinite[column]) for column in columns},
        "duplicate_rows_within_file": duplicate_rows,
        "constant_features": constants,
        "near_constant_features_at_99_percent": near_constants,
        "label_distribution": dict(sorted(label_counts.items())),
    }


def build_report(source_dir: Path, chunk_size: int = 50_000) -> dict[str, Any]:
    files = sorted(source_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {source_dir}")

    file_reports = [_scan_file(path, chunk_size) for path in files]
    all_columns = file_reports[0]["columns"]
    common_schema = all(report["columns"] == all_columns for report in file_reports)
    types: dict[str, set[str]] = {column: set() for column in all_columns}
    missing: Counter[str] = Counter()
    infinite: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    unique_values: dict[str, set[Any]] = {column: set() for column in all_columns}
    near_candidates: set[str] = set()
    duplicate_hashes: set[int] = set()
    cross_file_duplicate_rows = 0
    total_rows = 0

    for report in file_reports:
        total_rows += report["rows"]
        labels.update(report["label_distribution"])
        for column, values in report["data_types"].items():
            types[column].update(values)
        missing.update(report["missing_values"])
        infinite.update(report["infinite_values"])
        near_candidates.update(report["near_constant_features_at_99_percent"])

    for path in files:
        for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
            chunk.columns = _clean_columns(chunk.columns)
            hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy(dtype="uint64")
            for value in hashes:
                key = int(value)
                if key in duplicate_hashes:
                    cross_file_duplicate_rows += 1
                else:
                    duplicate_hashes.add(key)
            for column in all_columns:
                current = unique_values[column]
                if len(current) <= 1:
                    for value in chunk[column].dropna().unique():
                        current.add(value)
                        if len(current) == 2:
                            break

    global_near_counts: dict[str, Counter[Any]] = {column: Counter() for column in near_candidates}
    if near_candidates:
        for path in files:
            for chunk in pd.read_csv(
                path,
                chunksize=chunk_size,
                low_memory=False,
                usecols=lambda column: column.strip() in near_candidates,
            ):
                chunk.columns = _clean_columns(chunk.columns)
                for column in near_candidates:
                    global_near_counts[column].update(chunk[column].value_counts(dropna=False).to_dict())

    constants = {
        column: list(values) if values else []
        for column, values in unique_values.items()
        if len(values) <= 1
    }
    near_constants = {}
    for column, counts in global_near_counts.items():
        if counts:
            value, count = counts.most_common(1)[0]
            fraction = count / total_rows
            if fraction >= 0.99:
                near_constants[column] = {
                    "value": str(value),
                    "count": int(count),
                    "fraction": fraction,
                }

    numeric_features = [
        column for column in all_columns
        if column != "Label" and any(value.startswith(("int", "float")) for value in types[column])
    ]
    source_identifiers = [
        column for column in all_columns
        if any(token in column.lower() for token in ("flow id", "ip address", "timestamp", "src ip", "dst ip"))
    ]
    return {
        "source_dir": str(source_dir),
        "file_count": len(files),
        "total_rows": total_rows,
        "common_schema": common_schema,
        "files": file_reports,
        "dataset_summary": {
            "columns": all_columns,
            "data_types": {column: sorted(values) for column, values in types.items()},
            "missing_values": {column: int(missing[column]) for column in all_columns},
            "infinite_values": {column: int(infinite[column]) for column in all_columns},
            "duplicate_rows_across_combined_files": cross_file_duplicate_rows,
            "constant_features": constants,
            "near_constant_features_at_99_percent": near_constants,
            "identifier_or_metadata_columns": source_identifiers + (["Destination Port"] if "Destination Port" in all_columns else []),
            "target_column": "Label" if "Label" in all_columns else None,
            "numerical_features": numeric_features,
            "label_distribution": dict(sorted(labels.items())),
            "semantically_related_feature_families": {
                "flow_volume_and_rates": [c for c in numeric_features if any(k in c.lower() for k in ("packets", "bytes", "ratio")) and "length" not in c.lower()],
                "packet_sizes": [c for c in numeric_features if "length" in c.lower() or "packet size" in c.lower() or "segment size" in c.lower()],
                "tcp_connection_state": [c for c in numeric_features if "flag" in c.lower() or "header" in c.lower() or "win_bytes" in c.lower() or "seg_size" in c.lower() or "act_data" in c.lower()],
                "timing_and_bursts": [c for c in numeric_features if "iat" in c.lower() or "active" in c.lower() or "idle" in c.lower() or "duration" in c.lower()],
            },
        },
    }


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "dataset_inspection.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )

    lines = [
        "# CIC-IDS2017 Dataset Inspection",
        "",
        f"- Files: {report['file_count']}",
        f"- Rows: {report['total_rows']:,}",
        f"- Common schema: {report['common_schema']}",
        "",
        "## Per-file inventory",
        "",
        "| CSV | Size (bytes) | Rows | Columns | Exact duplicate rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for file_report in report["files"]:
        lines.append(
            f"| {file_report['file']} | {file_report['size_bytes']:,} | {file_report['rows']:,} "
            f"| {file_report['column_count']} | {file_report['duplicate_rows_within_file']:,} |"
        )

    summary = report["dataset_summary"]
    lines.extend([
        "",
        "## Dataset-wide findings",
        "",
        f"- Exact duplicate rows when files are combined: {summary['duplicate_rows_across_combined_files']:,}",
        f"- Identifier/metadata columns: {', '.join(summary['identifier_or_metadata_columns']) or 'None found'}",
        f"- Constant features: {', '.join(summary['constant_features']) or 'None'}",
        "- Near-constant features (one value has at least 99% of rows):",
    ])
    for column, detail in summary["near_constant_features_at_99_percent"].items():
        lines.append(f"  - {column}: {detail['value']} in {detail['count']:,} rows ({detail['fraction']:.4%})")
    lines.extend(["", "## Per-file details", ""])
    for file_report in report["files"]:
        lines.extend([
            f"### {file_report['file']}",
            "",
            f"Size: {file_report['size_bytes']:,} bytes; rows: {file_report['rows']:,}",
            "",
            "Columns, dtypes, missing and infinite values:",
            "",
            "| Column | dtype | missing | infinite |",
            "|---|---|---:|---:|",
        ])
        for column in file_report["columns"]:
            dtype = ", ".join(file_report["data_types"][column])
            lines.append(
                f"| {column} | {dtype} | {file_report['missing_values'][column]:,} "
                f"| {file_report['infinite_values'][column]:,} |"
            )
        lines.extend([
            "",
            f"Constant features: {json.dumps(file_report['constant_features'], ensure_ascii=False, default=_json_default)}",
            "",
            f"Near-constant features: {json.dumps(file_report['near_constant_features_at_99_percent'], ensure_ascii=False, default=_json_default)}",
            "",
            f"Label distribution: {json.dumps(file_report['label_distribution'], ensure_ascii=False, default=_json_default)}",
            "",
        ])
    lines.extend(["## Combined label distribution", "", "| Label | Rows |", "|---|---:|"])
    for label, count in summary["label_distribution"].items():
        lines.append(f"| {label} | {count:,} |")
    lines.extend(["", "## Numerical columns", "", ", ".join(summary["numerical_features"]), ""])
    for family, columns in summary["semantically_related_feature_families"].items():
        lines.extend([f"### {family.replace('_', ' ').title()}", "", ", ".join(columns), ""])
    (output_dir / "dataset_inspection.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {json_path} and {output_dir / 'dataset_inspection.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("MachineLearningCVE"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()
    write_report(build_report(args.source_dir, args.chunk_size), args.output_dir)


if __name__ == "__main__":
    main()
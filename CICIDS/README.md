# CIC-IDS2017 Multi-View Preparation

This repository prepares CIC-IDS2017 network-flow records as four semantically separated, independently normalized views for later unaligned multi-view clustering. It does **not** implement MvDOT or use labels to construct features.

## Project Layout

- `MachineLearningCVE/`: original source CSV files, read-only.
- `data/`: notes about the raw data location.
- `configs/views.json`: the authoritative view composition, seeds, and preprocessing decisions.
- `src/inspect_dataset.py`: full streaming dataset profile and report writer.
- `src/preprocess.py`: duplicate handling, cleanup, imputation, normalization, and view output.
- `src/create_alignment.py`: controlled seeded alignment permutations.
- `src/evaluate.py`: ACC, NMI, and ARI for externally generated predictions.
- `reports/`: generated dataset inspection in Markdown and JSON.
- `outputs/`: normalized view arrays, labels, metadata, and alignment maps.
- `tests/`: focused checks for composition, alignment, and metrics.

Install dependencies with `python -m pip install -r requirements.txt`.

## Run

```powershell
python -m src.inspect_dataset
python -m src.preprocess
python -m src.create_alignment
pytest
```

The inspector reads every CSV in chunks and writes `reports/dataset_inspection.md` and `reports/dataset_inspection.json`. Its reports contain file sizes, schemas, per-column types and integrity counts, duplicate rows, constants and near-constants, label counts, numeric features, and semantic feature families.

## View Composition

The 60 features are assigned once, with no feature shared between views:

| View | Meaning | Features |
|---|---|---:|
| `view_1_flow_traffic` | flow volume and aggregate rates | 7 |
| `view_2_packet` | packet and segment size distributions | 16 |
| `view_3_tcp_connection` | TCP flags, headers, windows, and data-segment behavior | 14 |
| `view_4_temporal` | flow, directional inter-arrival, active, and idle timing | 23 |

`configs/views.json` lists the exact names. Header whitespace is stripped on load; collisions after stripping fail rather than being guessed through.

## Preprocessing Decisions

1. **Identifiers and metadata:** This CIC-IDS2017 schema has no Flow ID, IP address, or timestamp columns. `Destination Port` is excluded because it is a nominal endpoint attribute rather than a continuous flow measurement. `Label` is stored separately and is excluded from every feature transformation.
2. **Irrelevant and redundant features:** Globally constant columns are excluded from the views. `Fwd Header Length.1` is omitted because it duplicates `Fwd Header Length`. Subflow packet/byte aggregates are omitted to avoid overlapping directional volume summaries; subflow packet counts duplicate the corresponding total packet counts in this scan, while byte aggregates are not guaranteed to be exact duplicates in every file. The six universally constant bulk features are absent. Rare TCP flag-count events remain because their sparsity can be informative.
3. **Duplicates:** Exact duplicate full rows are removed across the combined inputs, keeping the first record in sorted filename order. A pandas 64-bit row hash is used to track seen rows in memory. `metadata.json` records raw rows and removed duplicates.
4. **NaN and Inf:** Selected feature values are numerically coerced, and non-finite values become missing. Each retained feature is median-imputed; a feature that is entirely invalid is dropped and that change is reflected in output metadata.
5. **Normalization:** A separate `StandardScaler` is fitted for each view, after imputation. Scaling is unsupervised and never reads the label column. Saved matrices use `float32` to reduce storage.
6. **Labels and provenance:** `outputs/labels.csv.gz` contains the retained row index, original label, and source filename. Labels are only for later evaluation. `outputs/metadata.json` records view membership, imputation values, shapes, decisions, and output SHA-256 digests.
7. **Alignment:** All view matrices retain their common base row order. Each experiment stores a row-index mapping and a Boolean aligned-position mask for every view. The first configured view is the reference; each other view retains exactly `floor(N * fraction + 0.5)` same-index rows and deranges the rest. The saved datasets are reconstructed as `base_view[row_indices]`. This avoids duplicating the large matrices while preserving the exact experiment data and row-level alignment record. One fixed seed controls independent deterministic mappings. Requested levels are 100%, 75%, 50%, 25%, and 0%.

## Outputs

```text
outputs/
  metadata.json
  labels.csv.gz
  views/<view_name>.npy
  views/<view_name>.csv.gz
  alignment/experiments.json
  alignment/align_<percent>/manifest.json
  alignment/align_<percent>/<view_name>_row_indices.npy
  alignment/align_<percent>/<view_name>_aligned_positions.npy
```

Each view is saved in both efficient `.npy` format and readable `.csv.gz` format. The CSVs include a `row_index` column followed by that view's feature columns; pandas reads them directly with `pd.read_csv("outputs/views/view_1_flow_traffic.csv.gz")`. Labels remain separate in `outputs/labels.csv.gz` and join on `row_index`.

Load one aligned view with `np.load("outputs/views/view_1_flow_traffic.npy", mmap_mode="r")[row_indices]`, where `row_indices` is loaded from the selected experiment directory. For large outputs, keep `mmap_mode="r"` to avoid copying the base array into memory.

## Evaluation

Create a CSV with unique `row_index` and `cluster` columns, then run:

```powershell
python -m src.evaluate --predictions predictions.csv --output metrics.json
```

The evaluator joins predictions to the separately saved labels by row index and computes permutation-invariant clustering accuracy (Hungarian assignment), NMI, and ARI. It validates one-to-one row coverage; label values are not accepted by the feature-preparation scripts.
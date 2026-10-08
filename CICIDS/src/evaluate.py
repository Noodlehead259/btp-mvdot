"""Evaluate externally generated cluster predictions against held-out labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_values = LabelEncoder().fit_transform(y_true)
    predicted_values = LabelEncoder().fit_transform(y_pred)
    size = max(int(true_values.max()) + 1, int(predicted_values.max()) + 1)
    contingency = np.zeros((size, size), dtype=np.int64)
    np.add.at(contingency, (true_values, predicted_values), 1)
    rows, columns = linear_sum_assignment(contingency.max() - contingency)
    return float(contingency[rows, columns].sum() / len(true_values))


def evaluate(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    if len(labels) != len(predictions):
        raise ValueError("Labels and predictions must have the same number of rows.")
    if len(labels) == 0:
        raise ValueError("Cannot evaluate an empty set of predictions.")
    return {
        "ACC": clustering_accuracy(labels, predictions),
        "NMI": float(normalized_mutual_info_score(labels, predictions)),
        "ARI": float(adjusted_rand_score(labels, predictions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("outputs/labels.csv.gz"))
    parser.add_argument("--predictions", type=Path, required=True, help="CSV with row_index and cluster columns")
    parser.add_argument("--cluster-column", default="cluster")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    labels_frame = pd.read_csv(args.labels)
    predictions_frame = pd.read_csv(args.predictions)
    required_labels = {"row_index", "label"}
    required_predictions = {"row_index", args.cluster_column}
    if not required_labels.issubset(labels_frame.columns):
        raise ValueError(f"Labels file must contain {sorted(required_labels)}")
    if not required_predictions.issubset(predictions_frame.columns):
        raise ValueError(f"Predictions file must contain {sorted(required_predictions)}")
    merged = labels_frame[["row_index", "label"]].merge(
        predictions_frame[["row_index", args.cluster_column]],
        on="row_index",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(labels_frame) or len(merged) != len(predictions_frame):
        raise ValueError("Labels and predictions must cover the same unique row indices.")
    result = evaluate(
        merged["label"].astype(str).to_numpy(),
        merged[args.cluster_column].to_numpy(),
    )
    result["n_samples"] = int(len(merged))
    result["n_true_classes"] = int(merged["label"].nunique())
    result["n_predicted_clusters"] = int(merged[args.cluster_column].nunique())
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
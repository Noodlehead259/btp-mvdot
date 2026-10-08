"""Create reproducible row mappings for controlled view alignment experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _derangement(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(values) == 0:
        return values.copy()
    if len(values) < 2:
        raise ValueError("At least two rows are required to create an unaligned subset.")
    shuffled = rng.permutation(values)
    while np.any(shuffled == values):
        shuffled = rng.permutation(values)
    return shuffled


def build_alignment_map(
    row_count: int,
    alignment_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= alignment_fraction <= 1.0:
        raise ValueError("Alignment fraction must be between 0 and 1.")
    row_indices = np.arange(row_count, dtype=np.int64)
    aligned_count = int(np.floor(row_count * alignment_fraction + 0.5))
    aligned_positions = np.zeros(row_count, dtype=bool)
    if aligned_count:
        aligned_positions[rng.choice(row_count, size=aligned_count, replace=False)] = True
    unaligned_positions = np.flatnonzero(~aligned_positions)
    row_indices[unaligned_positions] = _derangement(unaligned_positions, rng)
    return row_indices, aligned_positions


def create_experiments(
    output_dir: Path,
    alignment_levels: list[float],
    random_seed: int,
    reference_view: str | None = None,
) -> list[dict[str, Any]]:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Run preprocessing first; missing {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    view_names = list(metadata["views"])
    reference_view = reference_view or view_names[0]
    if reference_view not in view_names:
        raise ValueError(f"Unknown reference view: {reference_view}")
    row_count = metadata["retained_row_count"]
    root_rng = np.random.default_rng(random_seed)
    experiment_records = []

    for level in alignment_levels:
        level_tag = f"align_{int(round(level * 100)):03d}"
        experiment_dir = output_dir / "alignment" / level_tag
        experiment_dir.mkdir(parents=True, exist_ok=True)
        view_records = {}
        for view_name in view_names:
            mapping_path = f"{view_name}_row_indices.npy"
            mask_path = f"{view_name}_aligned_positions.npy"
            if view_name == reference_view:
                row_indices = np.arange(row_count, dtype=np.int64)
                aligned_positions = np.ones(row_count, dtype=bool)
            else:
                rng = np.random.default_rng(root_rng.integers(0, np.iinfo(np.uint32).max))
                row_indices, aligned_positions = build_alignment_map(row_count, level, rng)
            np.save(experiment_dir / mapping_path, row_indices, allow_pickle=False)
            np.save(experiment_dir / mask_path, aligned_positions, allow_pickle=False)
            view_records[view_name] = {
                "base_array": metadata["views"][view_name]["file"],
                "row_indices": mapping_path,
                "aligned_positions": mask_path,
                "aligned_count": int(aligned_positions.sum()),
                "alignment_fraction": float(aligned_positions.mean()),
            }

        record = {
            "alignment_fraction_requested": level,
            "seed": random_seed,
            "row_count": row_count,
            "reference_view": reference_view,
            "views": view_records,
            "dataset_definition": "For a view, experiment rows equal base_view[row_indices]; compare index positions to the reference view. Labels are not used to create mappings.",
        }
        (experiment_dir / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        experiment_records.append(record)

    summary_path = output_dir / "alignment" / "experiments.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(experiment_records, indent=2), encoding="utf-8")
    print(f"Wrote {len(experiment_records)} alignment experiments to {summary_path.parent}")
    return experiment_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/views.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(config["output_dir"])
    create_experiments(output_dir, config["alignment_levels"], config["random_seed"])


if __name__ == "__main__":
    main()
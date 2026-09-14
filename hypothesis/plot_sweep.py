import csv
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

results_path = (
    project_root /
    "hypothesis" /
    "results" /
    "noise_sweep.csv"
)

output_path = project_root / "figures"

rows = []

with open(
    results_path,
    "r"
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        rows.append(row)

sigmas = [
    float(row["sigma"])
    for row in rows
]

disagreement = [
    float(row["disagreement_rate"]) * 100
    for row in rows
]

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    sigmas,
    disagreement,
    marker="o"
)

plt.xlabel(
    "gaussian noise standard deviation (σ)"
)

plt.ylabel(
    "cluster disagreement rate (%)"
)

plt.title(
    "cluster disagreement under increasing noise"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

output_file = (
    output_path /
    "figure_3_noise_sensitivity.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "saved:",
    output_file
)
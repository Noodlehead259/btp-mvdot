import csv
import numpy as np
import torch
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_idx_images(filename):
    with open(filename, "rb") as f:
        magic, num_images, rows, cols = np.frombuffer(
            f.read(16),
            dtype=">i4"
        )

        images = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

    return images.reshape(num_images, rows, cols)

def kmeans(data, k, iterations=100, seed=42):
    torch.manual_seed(seed)

    indices = torch.randperm(data.shape[0], device=device)[:k]
    centers = data[indices].clone()

    for _ in range(iterations):
        distances = (
            (data[:, None, :] - centers[None, :, :]) ** 2
        ).sum(dim=2)

        labels = torch.argmin(distances, dim=1)

        new_centers = torch.zeros_like(centers)

        for j in range(k):
            mask = labels == j

            if mask.any():
                new_centers[j] = data[mask].mean(dim=0)
            else:
                new_centers[j] = centers[j]

        if torch.allclose(centers, new_centers, atol=1e-5):
            centers = new_centers
            break

        centers = new_centers

    return labels, centers

def best_cluster_mapping(clean_labels, noisy_labels, k):
    overlap = np.zeros((k, k), dtype=np.int64)

    for i in range(k):
        for j in range(k):
            overlap[i, j] = np.sum(
                (clean_labels == i) &
                (noisy_labels == j)
            )

    dp = {0: (0, [])}

    for noisy_cluster in range(k):
        new_dp = {}

        for mask, (score, assignment) in dp.items():
            for clean_cluster in range(k):
                if mask & (1 << clean_cluster):
                    continue

                new_mask = mask | (1 << clean_cluster)
                new_score = score + overlap[clean_cluster, noisy_cluster]

                if (
                    new_mask not in new_dp or
                    new_score > new_dp[new_mask][0]
                ):
                    new_dp[new_mask] = (
                        new_score,
                        assignment + [clean_cluster]
                    )

        dp = new_dp

    best_mask = (1 << k) - 1
    best_assignment = dp[best_mask][1]

    mapping = {
        noisy_cluster: best_assignment[noisy_cluster]
        for noisy_cluster in range(k)
    }

    return mapping, overlap

def entropy(labels):
    counts = np.bincount(labels)
    probabilities = counts[counts > 0] / len(labels)

    return -np.sum(
        probabilities * np.log(probabilities)
    )

def nmi(labels_a, labels_b):
    n = len(labels_a)

    contingency = np.zeros(
        (labels_a.max() + 1, labels_b.max() + 1),
        dtype=np.int64
    )

    for a, b in zip(labels_a, labels_b):
        contingency[a, b] += 1

    mi = 0.0

    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            nij = contingency[i, j]

            if nij == 0:
                continue

            ni = contingency[i].sum()
            nj = contingency[:, j].sum()

            mi += (
                nij / n
            ) * np.log(
                (nij * n) / (ni * nj)
            )

    denominator = np.sqrt(
        entropy(labels_a) *
        entropy(labels_b)
    )

    if denominator == 0:
        return 1.0

    return mi / denominator

def ari(labels_a, labels_b):
    n = len(labels_a)

    contingency = np.zeros(
        (labels_a.max() + 1, labels_b.max() + 1),
        dtype=np.int64
    )

    for a, b in zip(labels_a, labels_b):
        contingency[a, b] += 1

    def combinations_2(x):
        return x * (x - 1) // 2

    index = np.sum(
        combinations_2(contingency)
    )

    row_sum = np.sum(
        combinations_2(contingency.sum(axis=1))
    )

    col_sum = np.sum(
        combinations_2(contingency.sum(axis=0))
    )

    total = combinations_2(n)

    expected = row_sum * col_sum / total

    maximum = (row_sum + col_sum) / 2

    if maximum == expected:
        return 1.0

    return (index - expected) / (maximum - expected)

def run_experiment(images, sigma):
    normalized_images = (
        images.astype(np.float32) / 255.0
    )

    clean_data = torch.from_numpy(
        normalized_images.reshape(-1, 784)
    ).to(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    noise = torch.randn(
        clean_data.shape,
        generator=generator,
        device=device
    ) * sigma

    noisy_data = torch.clamp(
        clean_data + noise,
        0.0,
        1.0
    )

    clean_labels, clean_centers = kmeans(
        clean_data,
        10,
        seed=42
    )

    noisy_labels, noisy_centers = kmeans(
        noisy_data,
        10,
        seed=42
    )

    clean_labels = clean_labels.cpu().numpy()
    noisy_labels = noisy_labels.cpu().numpy()

    mapping, overlap = best_cluster_mapping(
        clean_labels,
        noisy_labels,
        10
    )

    aligned_noisy_labels = np.array([
        mapping[label]
        for label in noisy_labels
    ])

    changed_indices = np.where(
        clean_labels != aligned_noisy_labels
    )[0]

    disagreement = (
        len(changed_indices) /
        len(images)
    )

    result = {
        "sigma": sigma,
        "total_images": len(images),
        "changed_images": len(changed_indices),
        "disagreement_rate": disagreement,
        "nmi": nmi(clean_labels, aligned_noisy_labels),
        "ari": ari(clean_labels, aligned_noisy_labels)
    }

    return result, clean_labels, aligned_noisy_labels, changed_indices

project_root = Path(__file__).resolve().parent.parent

images = read_idx_images(
    str(project_root / "data" / "train-images.idx3-ubyte")
)

print("device:", device)
print("gpu:", torch.cuda.get_device_name(0))
print("total images:", len(images))
print("starting experiment...")

sigma = 0.25

result, clean_labels, noisy_labels, changed_indices = run_experiment(
    images,
    sigma
)

print()
print("sigma:", result["sigma"])
print("total images:", result["total_images"])
print("changed images:", result["changed_images"])
print(
    "disagreement rate:",
    result["disagreement_rate"] * 100,
    "%"
)
print("nmi:", result["nmi"])
print("ari:", result["ari"])

np.savez_compressed(
    project_root / "hypothesis" / "results" / "labels_sigma_025.npz",
    clean_labels=clean_labels,
    noisy_labels=noisy_labels,
    changed_indices=changed_indices
)

with open(
    project_root / "hypothesis" / "results" / "metrics.csv",
    "w",
    newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sigma",
            "total_images",
            "changed_images",
            "disagreement_rate",
            "nmi",
            "ari"
        ]
    )

    writer.writeheader()
    writer.writerow(result)

print()
print("results saved.")
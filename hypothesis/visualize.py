import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

data_path = (
    project_root /
    "data" /
    "train-images.idx3-ubyte"
)

labels_path = (
    project_root /
    "hypothesis" /
    "results" /
    "labels_sigma_025.npz"
)

samples_path = (
    project_root /
    "hypothesis" /
    "results" /
    "changed_samples_sigma_025.npz"
)

output_path = project_root / "figures"

output_path.mkdir(exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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

    return images.reshape(
        num_images,
        rows,
        cols
    )

labels = np.load(labels_path)

clean_labels = labels["clean_labels"]
noisy_labels = labels["noisy_labels"]

samples = np.load(samples_path)

changed_indices = samples["changed_indices"]
changed_clean_images = samples["clean_images"]
changed_noisy_images = samples["noisy_images"]
changed_clean_labels = samples["clean_labels"]
changed_noisy_labels = samples["noisy_labels"]

images = read_idx_images(data_path)

clean_images = (
    images.astype(np.float32) / 255.0
)

clean_data = torch.from_numpy(
    clean_images.reshape(-1, 784)
).to(device)

generator = torch.Generator(
    device=device
)

generator.manual_seed(42)

base_noise = torch.randn(
    clean_data.shape,
    generator=generator,
    device=device
)

noisy_data = torch.clamp(
    clean_data + base_noise * 0.25,
    0.0,
    1.0
)

print("device:", device)
print("total images:", len(clean_images))
print("changed images:", len(changed_indices))

print()
print("creating figure 1...")

rng = np.random.default_rng(42)

sample_size = min(
    10000,
    len(clean_images)
)

sample_indices = rng.choice(
    len(clean_images),
    sample_size,
    replace=False
)

combined = torch.cat(
    [
        clean_data[sample_indices],
        noisy_data[sample_indices]
    ],
    dim=0
)

mean = combined.mean(
    dim=0,
    keepdim=True
)

combined = combined - mean

covariance = (
    combined.T @ combined
) / (combined.shape[0] - 1)

eigenvalues, eigenvectors = torch.linalg.eigh(
    covariance
)

components = eigenvectors[:, -2:]

clean_sample = (
    clean_data[sample_indices] - mean
)

noisy_sample = (
    noisy_data[sample_indices] - mean
)

clean_2d = (
    clean_sample @ components
).cpu().numpy()

noisy_2d = (
    noisy_sample @ components
).cpu().numpy()

clean_sample_labels = (
    clean_labels[sample_indices]
)

noisy_sample_labels = (
    noisy_labels[sample_indices]
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 7)
)

for cluster in range(10):
    mask = clean_sample_labels == cluster

    axes[0].scatter(
        clean_2d[mask, 0],
        clean_2d[mask, 1],
        s=5,
        alpha=0.35,
        label=f"cluster {cluster}"
    )

axes[0].set_title(
    "clean mnist"
)

axes[0].set_xlabel(
    "principal component 1"
)

axes[0].set_ylabel(
    "principal component 2"
)

for cluster in range(10):
    mask = noisy_sample_labels == cluster

    axes[1].scatter(
        noisy_2d[mask, 0],
        noisy_2d[mask, 1],
        s=5,
        alpha=0.35,
        label=f"cluster {cluster}"
    )

axes[1].set_title(
    "noisy mnist, sigma = 0.25"
)

axes[1].set_xlabel(
    "principal component 1"
)

axes[1].set_ylabel(
    "principal component 2"
)

handles, legend_labels = (
    axes[0].get_legend_handles_labels()
)

fig.legend(
    handles,
    legend_labels,
    loc="center right",
    bbox_to_anchor=(1.02, 0.5)
)

plt.tight_layout()

figure_1 = (
    output_path /
    "figure_1_cluster_structure.png"
)

plt.savefig(
    figure_1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "saved:",
    figure_1
)

print()
print("creating figure 2...")

selected = []

seen_transitions = set()

for i in range(len(changed_indices)):
    transition = (
        int(changed_clean_labels[i]),
        int(changed_noisy_labels[i])
    )

    if transition not in seen_transitions:
        selected.append(i)
        seen_transitions.add(transition)

    if len(selected) == 12:
        break

if len(selected) < 12:
    for i in range(len(changed_indices)):
        if i not in selected:
            selected.append(i)

        if len(selected) == 12:
            break

fig, axes = plt.subplots(
    12,
    3,
    figsize=(8, 24)
)

for row, position in enumerate(selected):
    axes[row, 0].imshow(
        changed_clean_images[position],
        cmap="gray"
    )

    axes[row, 0].set_title(
        f"sample {changed_indices[position]}\n"
        f"clean cluster: "
        f"{changed_clean_labels[position]}"
    )

    axes[row, 1].imshow(
        changed_noisy_images[position],
        cmap="gray"
    )

    axes[row, 1].set_title(
        f"sample {changed_indices[position]}\n"
        f"noisy cluster: "
        f"{changed_noisy_labels[position]}"
    )

    axes[row, 2].text(
        0.5,
        0.5,
        f"{changed_clean_labels[position]}"
        f" → "
        f"{changed_noisy_labels[position]}",
        ha="center",
        va="center",
        fontsize=16
    )

    axes[row, 0].axis("off")
    axes[row, 1].axis("off")
    axes[row, 2].axis("off")

plt.suptitle(
    "representative samples with changed cluster assignments",
    fontsize=16
)

plt.tight_layout()

figure_2 = (
    output_path /
    "figure_2_changed_samples.png"
)

plt.savefig(
    figure_2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "saved:",
    figure_2
)

print()
print("figures 1 and 2 complete.")
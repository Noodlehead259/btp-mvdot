import numpy as np

def read_idx_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    return images

def kmeans(data, k, iterations=100, seed=42):
    rng = np.random.default_rng(seed)

    indices = rng.choice(len(data), k, replace=False)
    centers = data[indices].copy()

    for _ in range(iterations):
        distances = np.sum(
            (data[:, None, :] - centers[None, :, :]) ** 2,
            axis=2
        )

        labels = np.argmin(distances, axis=1)

        new_centers = np.zeros_like(centers)

        for j in range(k):
            points = data[labels == j]

            if len(points) > 0:
                new_centers[j] = points.mean(axis=0)
            else:
                new_centers[j] = centers[j]

        if np.allclose(centers, new_centers):
            break

        centers = new_centers

    return labels, centers

def best_cluster_mapping(clean_labels, noisy_labels, k):
    overlap = np.zeros((k, k), dtype=int)

    for i in range(k):
        for j in range(k):
            overlap[i, j] = np.sum(
                (clean_labels == i) &
                (noisy_labels == j)
            )

    mapping = {}
    used_noisy = set()

    for _ in range(k):
        best_value = -1
        best_i = -1
        best_j = -1

        for i in range(k):
            if i in mapping.values():
                continue

            for j in range(k):
                if j in used_noisy:
                    continue

                if overlap[i, j] > best_value:
                    best_value = overlap[i, j]
                    best_i = i
                    best_j = j

        mapping[best_j] = best_i
        used_noisy.add(best_j)

    return mapping, overlap


images = read_idx_images('train-images.idx3-ubyte')

images = images[:100]

normalized_images = images.astype(np.float32) / 255.0

np.random.seed(42)

noise = np.random.normal(
    loc=0.0,
    scale=0.25,
    size=normalized_images.shape
)

noisy_images = np.clip(
    normalized_images + noise,
    0.0,
    1.0
)

clean_data = normalized_images.reshape(100, 784)
noisy_data = noisy_images.reshape(100, 784)

k = 10

clean_labels, clean_centers = kmeans(
    clean_data,
    k
)

noisy_labels, noisy_centers = kmeans(
    noisy_data,
    k
)

mapping, overlap = best_cluster_mapping(
    clean_labels,
    noisy_labels,
    k
)

aligned_noisy_labels = np.array([
    mapping[label]
    for label in noisy_labels
])

changed_indices = np.where(
    clean_labels != aligned_noisy_labels
)[0]

print("total images:", len(images))
print("changed images:", len(changed_indices))
print(
    "change percentage:",
    len(changed_indices) / len(images) * 100
)

print("\ncluster mapping:")
for noisy_cluster, clean_cluster in mapping.items():
    print(
        "noisy cluster",
        noisy_cluster,
        "-> clean cluster",
        clean_cluster
    )

print("\nchanged datapoints:")

for i in changed_indices:
    print(
        "index:", i,
        "clean cluster:", clean_labels[i],
        "noisy cluster:", aligned_noisy_labels[i]
    )
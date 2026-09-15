import struct
import numpy as np


def load_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">iiii", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)

    images = images.reshape(n, rows * cols)
    return images.astype(np.float32) / 255.0


def load_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">ii", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels.astype(np.int64)


def create_noisy_view(images, sigma=0.25, seed=42):
    rng = np.random.default_rng(seed)

    noise = rng.normal(
        0.0,
        sigma,
        size=images.shape
    ).astype(np.float32)

    noisy = images + noise
    noisy = np.clip(noisy, 0.0, 1.0)

    return noisy


def load_mnist():
    train_images = load_images("data/train-images.idx3-ubyte")
    train_labels = load_labels("data/train-labels.idx1-ubyte")

    test_images = load_images("data/t10k-images.idx3-ubyte")
    test_labels = load_labels("data/t10k-labels.idx1-ubyte")

    images = np.concatenate(
        [train_images, test_images],
        axis=0
    )

    labels = np.concatenate(
        [train_labels, test_labels],
        axis=0
    )

    return images, labels
import torch
from torch.utils.data import Dataset
import numpy as np


class noisy_mnist(Dataset):
    def __init__(self, view1, view2, labels, aligned_rate=1.0, seed=42):
        self.view1 = torch.tensor(view1, dtype=torch.float32)
        self.view2 = torch.tensor(view2, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        if aligned_rate < 1.0:
            rng = np.random.default_rng(seed)

            n = len(self.view2)
            n_aligned = int(n * aligned_rate)

            aligned_indices = np.arange(n_aligned)
            unaligned_indices = np.arange(n_aligned, n)

            shuffled = unaligned_indices.copy()
            rng.shuffle(shuffled)

            permutation = np.concatenate([
                aligned_indices,
                shuffled
            ])

            self.view2 = self.view2[permutation]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.view1[index],
            self.view2[index],
            self.labels[index]
        )
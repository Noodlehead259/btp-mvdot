import numpy as np
import torch
from torch.utils.data import Dataset


class noisy_mnist(Dataset):
    def __init__(
        self,
        view1,
        view2,
        labels,
        aligned_rate=1.0,
        seed=42
    ):
        self.view1 = torch.as_tensor(
            view1,
            dtype=torch.float32
        )

        self.view2 = torch.as_tensor(
            view2,
            dtype=torch.float32
        )

        self.labels = torch.as_tensor(
            labels,
            dtype=torch.long
        )

        if not 0.0 <= aligned_rate <= 1.0:
            raise ValueError(
                "aligned_rate must be between 0 and 1"
            )

        if len(self.view1) != len(self.view2):
            raise ValueError(
                "views must contain the same number of samples"
            )

        if len(self.view1) != len(self.labels):
            raise ValueError(
                "views and labels must contain the same number of samples"
            )

        n = len(self.view2)
        aligned_count = int(
            n * aligned_rate
        )

        rng = np.random.default_rng(seed)

        permutation = np.arange(n)

        if aligned_count < n:
            unaligned = np.arange(
                aligned_count,
                n
            )

            shuffled = unaligned.copy()

            rng.shuffle(
                shuffled
            )

            permutation[
                aligned_count:
            ] = shuffled

        self.view2 = self.view2[
            torch.from_numpy(permutation)
        ]

        self.aligned_rate = aligned_rate

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.view1[index],
            self.view2[index],
            self.labels[index]
        )
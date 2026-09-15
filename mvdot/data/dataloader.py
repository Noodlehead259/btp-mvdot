import torch
from torch.utils.data import DataLoader, TensorDataset


def create_dataloader(
    view1,
    view2,
    labels,
    batch_size=256,
    shuffle=True
):
    dataset = TensorDataset(
        view1,
        view2,
        labels
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True
    )
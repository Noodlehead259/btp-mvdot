import sys
import os
import torch

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )
)

from mvdot.data.preprocessing import (
    load_images,
    load_labels,
    create_noisy_view
)

from mvdot.data.dataset import noisy_mnist
from mvdot.data.dataloader import create_dataloader


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

images = load_images(
    "data/train-images.idx3-ubyte"
)

labels = load_labels(
    "data/train-labels.idx1-ubyte"
)

noisy_images = create_noisy_view(
    images,
    sigma=0.25
)

dataset = noisy_mnist(
    images,
    noisy_images,
    labels,
    aligned_rate=0.5
)

view1 = dataset.view1
view2 = dataset.view2
labels = dataset.labels

loader = create_dataloader(
    view1,
    view2,
    labels,
    batch_size=256,
    shuffle=True
)

print("device:", device)
print("total samples:", len(dataset))
print("number of batches:", len(loader))

for batch_index, batch in enumerate(loader):
    batch_view1, batch_view2, batch_labels = batch

    print(
        "batch:",
        batch_index + 1,
        "view 1:",
        batch_view1.shape,
        "view 2:",
        batch_view2.shape,
        "labels:",
        batch_labels.shape
    )

    if batch_index == 2:
        break

print()
print("batch pipeline test passed")
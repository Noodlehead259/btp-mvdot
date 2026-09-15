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

from mvdot.data.preprocessing import load_mnist, create_noisy_view
from mvdot.data.dataset import noisy_mnist


images, labels = load_mnist()

noisy_images = create_noisy_view(
    images,
    sigma=0.25
)

data_100 = noisy_mnist(
    images,
    noisy_images,
    labels,
    aligned_rate=1.0
)

data_50 = noisy_mnist(
    images,
    noisy_images,
    labels,
    aligned_rate=0.5
)

data_0 = noisy_mnist(
    images,
    noisy_images,
    labels,
    aligned_rate=0.0
)


print("number of samples:", len(data_100))
print("view 1 shape:", data_100.view1.shape)
print("view 2 shape:", data_100.view2.shape)
print("labels shape:", data_100.labels.shape)

print()
print("aligned rate 1.0")
print(
    "first sample same:",
    torch.allclose(
        data_100.view2[0],
        torch.tensor(noisy_images[0])
    )
)

print()
print("aligned rate 0.5")
print(
    "first 35000 samples preserved:",
    torch.equal(
        data_50.view2[:35000],
        torch.tensor(noisy_images[:35000])
    )
)

print()
print("aligned rate 0.0")
print(
    "first sample same:",
    torch.allclose(
        data_0.view2[0],
        torch.tensor(noisy_images[0])
    )
)
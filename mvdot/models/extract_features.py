import sys
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

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
from mvdot.models.autoencoder import autoencoder
from mvdot.models.barycenter import barycenter
from mvdot.ot.barycenter_ot import initialize_barycenter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device:", device)

images, labels = load_mnist()

noisy_images = create_noisy_view(
    images,
    sigma=0.25
)

view1 = torch.tensor(
    images,
    dtype=torch.float32
)

view2 = torch.tensor(
    noisy_images,
    dtype=torch.float32
)

dataset1 = TensorDataset(view1)
dataset2 = TensorDataset(view2)

loader1 = DataLoader(
    dataset1,
    batch_size=1024,
    shuffle=False,
    pin_memory=True
)

loader2 = DataLoader(
    dataset2,
    batch_size=1024,
    shuffle=False,
    pin_memory=True
)

model1 = autoencoder().to(device)
model2 = autoencoder().to(device)

model1.load_state_dict(
    torch.load(
        "mvdot/models/autoencoder_view1.pt",
        map_location=device,
        weights_only=True
    )
)

model2.load_state_dict(
    torch.load(
        "mvdot/models/autoencoder_view2.pt",
        map_location=device,
        weights_only=True
    )
)

model1.eval()
model2.eval()

features1 = []
features2 = []

with torch.no_grad():
    for (x,) in loader1:
        x = x.to(device, non_blocking=True)

        z = model1.encode(x)

        features1.append(
            z.cpu()
        )

    for (x,) in loader2:
        x = x.to(device, non_blocking=True)

        z = model2.encode(x)

        features2.append(
            z.cpu()
        )

z1 = torch.cat(features1, dim=0)
z2 = torch.cat(features2, dim=0)

print("z1 shape:", z1.shape)
print("z2 shape:", z2.shape)

z1_gpu = z1.to(device)
z2_gpu = z2.to(device)

centers, weights = initialize_barycenter(
    z1_gpu,
    z2_gpu,
    num_clusters=10,
    iterations=20,
    seed=42
)

print("barycenter centers shape:", centers.shape)
print("barycenter weights shape:", weights.shape)

print(
    "center norms:",
    torch.norm(centers, dim=1)
)

model = barycenter(
    num_clusters=10,
    feature_dim=128
).to(device)

with torch.no_grad():
    model.centers.copy_(centers)
    model.weights.copy_(weights)

torch.save(
    model.state_dict(),
    "mvdot/models/barycenter_init.pt"
)

print("barycenter initialization saved")
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

from mvdot.models.barycenter import barycenter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = barycenter(
    num_clusters=10,
    feature_dim=128
).to(device)

centers, weights = model()

print("device:", device)
print("centers shape:", centers.shape)
print("weights shape:", weights.shape)
print("weights:", weights)
print("centers require grad:", centers.requires_grad)
print(
    "center norms:",
    torch.norm(centers, dim=1)
)
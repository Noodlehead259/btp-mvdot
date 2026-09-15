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

from mvdot.ot.barycenter_ot import spherical_kmeans


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(
    10000,
    128,
    device=device
)

centers = spherical_kmeans(
    x,
    num_clusters=10,
    iterations=20
)

print("device:", device)
print("input shape:", x.shape)
print("centers shape:", centers.shape)
print("center norms:", torch.norm(centers, dim=1))
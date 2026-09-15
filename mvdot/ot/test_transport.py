import sys
import os
import torch
import torch.nn.functional as f

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )
)

from mvdot.ot.transport import sample_to_cluster_transport


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

n = 1000
k = 10
d = 128

features = torch.randn(
    n,
    d,
    device=device
)

centers = torch.randn(
    k,
    d,
    device=device
)

centers = f.normalize(
    centers,
    p=2,
    dim=1
)

weights = torch.ones(
    k,
    device=device
) / k

transport, cost = sample_to_cluster_transport(
    features,
    centers,
    weights,
    epsilon=0.1,
    iterations=100
)

source_mass = torch.ones(
    n,
    device=device
) / n

print("device:", device)
print("features shape:", features.shape)
print("centers shape:", centers.shape)
print("cost shape:", cost.shape)
print("transport shape:", transport.shape)

print(
    "source marginal error:",
    torch.max(
        torch.abs(
            transport.sum(dim=1) - source_mass
        )
    ).item()
)

print(
    "target marginal error:",
    torch.max(
        torch.abs(
            transport.sum(dim=0) - weights
        )
    ).item()
)

print(
    "total transport mass:",
    transport.sum().item()
)
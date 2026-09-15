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

from mvdot.ot.barycenter_ot import (
    initialize_barycenter,
    update_barycenter
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

n = 1000
d = 128
k = 10

z1 = torch.randn(
    n,
    d,
    device=device
)

z2 = torch.randn(
    n,
    d,
    device=device
)

z1 = f.normalize(
    z1,
    p=2,
    dim=1
)

z2 = f.normalize(
    z2,
    p=2,
    dim=1
)

centers, weights = initialize_barycenter(
    z1,
    z2,
    num_clusters=k,
    iterations=10
)

transport1 = torch.rand(
    n,
    k,
    device=device
)

transport2 = torch.rand(
    n,
    k,
    device=device
)

transport1 = (
    transport1
    / transport1.sum(dim=1, keepdim=True)
    / n
)

transport2 = (
    transport2
    / transport2.sum(dim=1, keepdim=True)
    / n
)

new_centers, new_weights = update_barycenter(
    [z1, z2],
    [transport1, transport2],
    centers,
    weights,
    alpha=0.98
)

print("device:", device)
print("old centers shape:", centers.shape)
print("new centers shape:", new_centers.shape)
print("old weights shape:", weights.shape)
print("new weights shape:", new_weights.shape)

print()
print(
    "new center norms:",
    torch.norm(
        new_centers,
        dim=1
    )
)

print()
print(
    "new weights:",
    new_weights
)

print(
    "weight sum:",
    new_weights.sum().item()
)

print(
    "center norm error:",
    torch.max(
        torch.abs(
            torch.norm(
                new_centers,
                dim=1
            ) - 1.0
        )
    ).item()
)
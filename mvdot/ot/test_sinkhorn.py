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

from mvdot.ot.sinkhorn import sinkhorn


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

n = 100
k = 10

cost = torch.rand(
    n,
    k,
    device=device
)

source_mass = torch.ones(
    n,
    device=device
) / n

target_mass = torch.ones(
    k,
    device=device
) / k

transport = sinkhorn(
    cost,
    source_mass,
    target_mass,
    epsilon=0.1,
    iterations=100
)

print("device:", device)
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
            transport.sum(dim=0) - target_mass
        )
    ).item()
)

print(
    "total transport mass:",
    transport.sum().item()
)
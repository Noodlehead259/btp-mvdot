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

from mvdot.losses.semantic import (
    cluster_probabilities,
    semantic_cost,
    view_weights,
    total_semantic_matching_loss
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

n = 100
d = 128
k = 10

features1 = torch.randn(
    n,
    d,
    device=device
)

features2 = torch.randn(
    n,
    d,
    device=device
)

centers = torch.randn(
    k,
    d,
    device=device
)

probabilities1 = cluster_probabilities(
    features1,
    centers
)

probabilities2 = cluster_probabilities(
    features2,
    centers
)

transport12 = torch.rand(
    n,
    n,
    device=device
)

transport21 = torch.rand(
    n,
    n,
    device=device
)

probabilities = [
    probabilities1,
    probabilities2
]

transports = {
    (0, 1): transport12,
    (1, 0): transport21
}

transport_costs = [
    torch.tensor(
        2.0,
        device=device
    ),
    torch.tensor(
        3.0,
        device=device
    )
]

cost = semantic_cost(
    probabilities1,
    probabilities2
)

weights = view_weights(
    transport_costs
)

loss, weights = total_semantic_matching_loss(
    transports,
    probabilities,
    transport_costs
)

print("device:", device)
print(
    "probabilities 1 shape:",
    probabilities1.shape
)
print(
    "probabilities 2 shape:",
    probabilities2.shape
)
print(
    "semantic cost shape:",
    cost.shape
)

print()
print("view weights:", weights)
print(
    "weight sum:",
    weights.sum().item()
)

print()
print("total semantic loss:", loss.item())

print()
print(
    "probability row sums:",
    probabilities1.sum(dim=1)[:5]
)

print(
    "minimum semantic cost:",
    cost.min().item()
)

print(
    "maximum semantic cost:",
    cost.max().item()
)
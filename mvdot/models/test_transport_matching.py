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

from mvdot.models.transport_matching import (
    cross_view_affinity,
    build_bipartite_graph,
    bfs_topology,
    topology_cost,
    semantic_cost_from_probabilities,
    cross_view_transport
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

probabilities1 = torch.softmax(
    torch.randn(n, k, device=device),
    dim=1
)

probabilities2 = torch.softmax(
    torch.randn(n, k, device=device),
    dim=1
)

edges, values = cross_view_affinity(
    features1,
    features2,
    centers,
    top_k=5,
    chunk_size=25
)

graph_edges = build_bipartite_graph(
    edges,
    n,
    n
)

topology = bfs_topology(
    graph_edges,
    n,
    n
).to(device)

topology_cost_matrix = topology_cost(
    topology
)

semantic_cost = semantic_cost_from_probabilities(
    probabilities1,
    probabilities2
)

transport, cost = cross_view_transport(
    semantic_cost,
    topology_cost_matrix,
    epsilon=0.1,
    iterations=100
)

source_mass = torch.ones(
    n,
    device=device
) / n

target_mass = torch.ones(
    n,
    device=device
) / n

print("device:", device)
print("semantic cost shape:", semantic_cost.shape)
print("topology cost shape:", topology_cost_matrix.shape)
print("hybrid cost shape:", cost.shape)
print("cross-view transport shape:", transport.shape)

print()

print(
    "source marginal error:",
    torch.max(
        torch.abs(
            transport.sum(dim=1)
            - source_mass
        )
    ).item()
)

print(
    "target marginal error:",
    torch.max(
        torch.abs(
            transport.sum(dim=0)
            - target_mass
        )
    ).item()
)

print(
    "total transport mass:",
    transport.sum().item()
)

print(
    "minimum hybrid cost:",
    cost.min().item()
)

print(
    "maximum hybrid cost:",
    cost.max().item()
)
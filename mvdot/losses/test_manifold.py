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

from mvdot.losses.manifold import (
    knn_graph,
    graph_laplacian,
    manifold_regularization,
    manifold_matching_loss
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

n = 500
d = 128
k = 10

z = torch.randn(
    n,
    d,
    device=device
)

transport = torch.rand(
    n,
    10,
    device=device
)

cost = torch.rand(
    n,
    10,
    device=device
)

graph = knn_graph(
    z,
    k=k,
    chunk_size=128
)

degree = graph_laplacian(
    graph
)

sparse_graph_cost = manifold_regularization(
    z,
    graph,
    degree
)

dense_graph = torch.zeros(
    n,
    n,
    device=device
)

indices = graph.indices()
values = graph.values()

dense_graph[
    indices[0],
    indices[1]
] = values

dense_degree = dense_graph.sum(
    dim=1
)

dense_laplacian = (
    torch.diag(dense_degree)
    - dense_graph
)

dense_graph_cost = torch.trace(
    z.t()
    @ dense_laplacian
    @ z
)

loss, transport_cost, graph_cost = manifold_matching_loss(
    transport,
    cost,
    z,
    graph,
    degree
)

print("device:", device)
print("z shape:", z.shape)
print("graph shape:", graph.shape)
print("number of graph edges:", graph._nnz())
print("degree shape:", degree.shape)

print()
print("sparse graph cost:", sparse_graph_cost.item())
print("dense graph cost:", dense_graph_cost.item())

print()
print(
    "absolute difference:",
    abs(
        sparse_graph_cost.item()
        - dense_graph_cost.item()
    )
)

print()
print("manifold loss:", loss.item())
print("transport cost:", transport_cost.item())
print("graph cost:", graph_cost.item())
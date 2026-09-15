import sys
import os
import torch

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from mvdot.models.autoencoder import autoencoder
from mvdot.ot.barycenter_ot import (
    initialize_barycenter,
    update_barycenter
)
from mvdot.ot.transport import sample_to_cluster_transport
from mvdot.losses.reconstruction import reconstruction_loss
from mvdot.losses.manifold import (
    knn_graph,
    graph_laplacian,
    manifold_matching_loss
)
from mvdot.models.transport_matching import (
    cross_view_affinity,
    build_bipartite_graph,
    bfs_topology,
    topology_cost,
    semantic_cost_from_probabilities,
    cross_view_transport
)
from mvdot.losses.semantic import (
    cluster_probabilities,
    total_semantic_matching_loss
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

n = 100
input_dim = 784
hidden_dim = 512
latent_dim = 128
num_clusters = 10

view1 = torch.randn(
    n,
    input_dim,
    device=device
)

view2 = torch.randn(
    n,
    input_dim,
    device=device
)

model1 = autoencoder(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    latent_dim=latent_dim
).to(device)

model2 = autoencoder(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    latent_dim=latent_dim
).to(device)

optimizer = torch.optim.Adam(
    list(model1.parameters())
    + list(model2.parameters()),
    lr=1e-4
)

criterion = reconstruction_loss()

with torch.no_grad():
    z1 = model1.encoder(view1)
    z2 = model2.encoder(view2)

    centers, weights = initialize_barycenter(
        z1,
        z2,
        num_clusters=num_clusters,
        iterations=10
    )

centers = centers.detach()
weights = weights.detach()

for epoch in range(3):
    optimizer.zero_grad()

    z1 = model1.encoder(view1)
    z2 = model2.encoder(view2)

    reconstruction1 = model1.decoder(z1)
    reconstruction2 = model2.decoder(z2)

    lrec1 = criterion(
        view1,
        reconstruction1
    )

    lrec2 = criterion(
        view2,
        reconstruction2
    )

    lrec = lrec1 + lrec2

    t1, cost1 = sample_to_cluster_transport(
        z1,
        centers,
        weights,
        epsilon=0.1,
        iterations=100
    )

    t2, cost2 = sample_to_cluster_transport(
        z2,
        centers,
        weights,
        epsilon=0.1,
        iterations=100
    )

    graph1 = knn_graph(
        z1,
        k=10,
        chunk_size=50
    )

    graph2 = knn_graph(
        z2,
        k=10,
        chunk_size=50
    )

    degree1 = graph_laplacian(
        graph1
    )

    degree2 = graph_laplacian(
        graph2
    )

    manifold1 = manifold_matching_loss(
        t1,
        cost1,
        z1,
        graph1,
        degree1
    )

    manifold2 = manifold_matching_loss(
        t2,
        cost2,
        z2,
        graph2,
        degree2
    )

    transport_cost1 = manifold1[1]
    graph_cost1 = manifold1[2]

    transport_cost2 = manifold2[1]
    graph_cost2 = manifold2[2]

    lmm = (
        transport_cost1
        + transport_cost2
        + 0.05 * (
            graph_cost1
            + graph_cost2
        )
    )

    probabilities1 = cluster_probabilities(
        z1,
        centers
    )

    probabilities2 = cluster_probabilities(
        z2,
        centers
    )

    edges, values = cross_view_affinity(
        z1,
        z2,
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

    semantic_cost_matrix = (
        semantic_cost_from_probabilities(
            probabilities1,
            probabilities2
        )
    )

    p12, hybrid_cost = cross_view_transport(
        semantic_cost_matrix,
        topology_cost_matrix,
        epsilon=0.1,
        iterations=100
    )

    p21 = p12.t()

    transports = {
        (0, 1): p12,
        (1, 0): p21
    }

    probabilities = [
        probabilities1,
        probabilities2
    ]

    transport_costs = [
        transport_cost1.detach(),
        transport_cost2.detach()
    ]

    lsm, view_weights = (
        total_semantic_matching_loss(
            transports,
            probabilities,
            transport_costs
        )
    )

    loss = (
        lrec
        + lmm
        + lsm
    )

    loss.backward()

    optimizer.step()

    with torch.no_grad():
        centers, weights = update_barycenter(
            [z1.detach(), z2.detach()],
            [t1.detach(), t2.detach()],
            centers,
            weights,
            alpha=0.98
        )

    print(
        "epoch:",
        epoch + 1,
        "lrec:",
        lrec.item(),
        "lmm:",
        lmm.item(),
        "lsm:",
        lsm.item(),
        "total:",
        loss.item()
    )

    print(
        "view weights:",
        view_weights
    )

print()
print("training integration test passed")
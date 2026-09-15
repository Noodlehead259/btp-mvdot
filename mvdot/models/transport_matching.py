import torch

from mvdot.ot.sinkhorn import sinkhorn


def barycenter_affinity(
    features,
    centers,
    sigma=1.0
):
    features_norm = (
        features * features
    ).sum(
        dim=1,
        keepdim=True
    )

    centers_norm = (
        centers * centers
    ).sum(
        dim=1,
        keepdim=True
    ).t()

    distance = (
        features_norm
        + centers_norm
        - 2.0 * (
            features @ centers.t()
        )
    )

    distance = torch.clamp(
        distance,
        min=0.0
    )

    return torch.exp(
        -distance
        / (
            2.0 * sigma * sigma
        )
    )


def cross_view_affinity(
    features1,
    features2,
    centers,
    sigma=1.0,
    top_k=5
):
    affinity1 = barycenter_affinity(
        features1,
        centers,
        sigma
    )

    affinity2 = barycenter_affinity(
        features2,
        centers,
        sigma
    )

    cross_affinity = (
        affinity1
        @ affinity2.t()
    )

    k = min(
        top_k,
        features2.size(0)
    )

    values, indices = torch.topk(
        cross_affinity,
        k=k,
        dim=1
    )

    rows = torch.arange(
        features1.size(0),
        device=features1.device
    ).unsqueeze(1)

    rows = rows.expand_as(
        indices
    )

    edges = torch.stack(
        [
            rows.reshape(-1),
            indices.reshape(-1)
        ],
        dim=0
    )

    return edges, values.reshape(-1)


def build_bipartite_graph(
    edges,
    n1,
    n2
):
    source = edges[0]

    target = (
        edges[1]
        + n1
    )

    forward = torch.stack(
        [
            source,
            target
        ],
        dim=0
    )

    backward = torch.stack(
        [
            target,
            source
        ],
        dim=0
    )

    return torch.cat(
        [
            forward,
            backward
        ],
        dim=1
    )


def bfs_topology(
    edges,
    n1,
    n2
):
    total_nodes = n1 + n2

    adjacency = [
        []
        for _ in range(
            total_nodes
        )
    ]

    edge_cpu = edges.detach().cpu()

    for i in range(
        edge_cpu.size(1)
    ):
        source = int(
            edge_cpu[0, i]
        )

        target = int(
            edge_cpu[1, i]
        )

        adjacency[
            source
        ].append(
            target
        )

    from collections import deque

    topology = torch.full(
        (n1, n2),
        float("inf")
    )

    for source in range(
        n1
    ):
        distance = [
            -1
        ] * total_nodes

        distance[source] = 0

        queue = deque(
            [source]
        )

        while queue:
            current = queue.popleft()

            for neighbor in adjacency[
                current
            ]:
                if distance[
                    neighbor
                ] != -1:
                    continue

                distance[
                    neighbor
                ] = (
                    distance[current]
                    + 1
                )

                queue.append(
                    neighbor
                )

        for target in range(
            n2
        ):
            node = n1 + target

            if distance[node] != -1:
                topology[
                    source,
                    target
                ] = distance[node]

    return topology


def topology_cost(
    topology
):
    cost = torch.ones_like(
        topology
    )

    finite = torch.isfinite(
        topology
    )

    cost[finite] = torch.sigmoid(
        topology[finite]
    )

    return cost


def semantic_cost_from_probabilities(
    probabilities1,
    probabilities2
):
    compatibility = (
        probabilities1
        @ probabilities2.t()
    )

    compatibility = torch.clamp(
        compatibility,
        min=1e-12
    )

    return -torch.log(
        compatibility
    )


def hybrid_cost(
    semantic_cost,
    topology_cost_matrix
):
    return (
        semantic_cost
        * topology_cost_matrix
    )


def cross_view_transport(
    semantic_cost,
    topology_cost_matrix,
    epsilon=0.1,
    iterations=100
):
    cost = hybrid_cost(
        semantic_cost,
        topology_cost_matrix
    )

    n1 = cost.size(0)
    n2 = cost.size(1)

    source_mass = torch.ones(
        n1,
        device=cost.device,
        dtype=cost.dtype
    ) / n1

    target_mass = torch.ones(
        n2,
        device=cost.device,
        dtype=cost.dtype
    ) / n2

    transport = sinkhorn(
        cost,
        source_mass,
        target_mass,
        epsilon=epsilon,
        iterations=iterations
    )

    return (
        transport,
        cost
    )
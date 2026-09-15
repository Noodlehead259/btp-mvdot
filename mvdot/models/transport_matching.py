import torch


def barycenter_affinity(
    features,
    centers,
    sigma=1.0
):
    features_norm = (features * features).sum(
        dim=1,
        keepdim=True
    )

    centers_norm = (centers * centers).sum(
        dim=1,
        keepdim=True
    ).t()

    distance = (
        features_norm
        + centers_norm
        - 2.0 * (features @ centers.t())
    )

    distance = torch.clamp(
        distance,
        min=0.0
    )

    affinity = torch.exp(
        -distance / (2.0 * sigma * sigma)
    )

    return affinity


def cross_view_affinity(
    features1,
    features2,
    centers,
    sigma=1.0,
    top_k=10,
    chunk_size=512
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

    n1 = features1.size(0)
    n2 = features2.size(0)

    rows = []
    cols = []
    values = []

    affinity2_t = affinity2.t()

    for start in range(0, n1, chunk_size):
        end = min(
            start + chunk_size,
            n1
        )

        cross_affinity = (
            affinity1[start:end]
            @ affinity2_t
        )

        k = min(
            top_k,
            n2
        )

        top_values, top_indices = torch.topk(
            cross_affinity,
            k=k,
            dim=1
        )

        row_indices = torch.arange(
            start,
            end,
            device=features1.device
        ).unsqueeze(1).expand_as(
            top_indices
        )

        rows.append(
            row_indices.reshape(-1)
        )

        cols.append(
            top_indices.reshape(-1)
        )

        values.append(
            top_values.reshape(-1)
        )

    rows = torch.cat(rows)
    cols = torch.cat(cols)
    values = torch.cat(values)

    edges = torch.stack(
        [rows, cols],
        dim=0
    )

    return edges, values


def build_bipartite_graph(
    edges,
    n1,
    n2
):
    source = edges[0]
    target = edges[1] + n1

    forward = torch.stack(
        [source, target],
        dim=0
    )

    backward = torch.stack(
        [target, source],
        dim=0
    )

    graph_edges = torch.cat(
        [forward, backward],
        dim=1
    )

    return graph_edges


def bfs_topology(
    edges,
    n1,
    n2,
    max_hops=None
):
    total_nodes = n1 + n2

    adjacency = [
        [] for _ in range(total_nodes)
    ]

    edge_cpu = edges.cpu()

    for i in range(edge_cpu.size(1)):
        source = edge_cpu[0, i].item()
        target = edge_cpu[1, i].item()

        adjacency[source].append(target)
        adjacency[target].append(source)

    topology = torch.full(
        (n1, n2),
        float("inf")
    )

    from collections import deque

    for source in range(n1):
        distance = [-1] * total_nodes

        distance[source] = 0

        queue = deque([source])

        while queue:
            current = queue.popleft()

            if max_hops is not None:
                if distance[current] >= max_hops:
                    continue

            for neighbor in adjacency[current]:
                if distance[neighbor] != -1:
                    continue

                distance[neighbor] = (
                    distance[current] + 1
                )

                queue.append(neighbor)

        for target in range(n2):
            target_node = n1 + target

            if distance[target_node] != -1:
                topology[source, target] = distance[
                    target_node
                ]

    return topology


def topology_cost(
    topology,
    scale=1.0
):
    cost = torch.ones_like(
        topology
    )

    finite = torch.isfinite(topology)

    cost[finite] = torch.sigmoid(
        topology[finite] / scale
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
    topology_cost
):
    return (
        semantic_cost
        * topology_cost
    )


def cross_view_transport(
    semantic_cost,
    topology_cost,
    epsilon=0.1,
    iterations=100
):
    from mvdot.ot.sinkhorn import sinkhorn

    n1 = semantic_cost.size(0)
    n2 = semantic_cost.size(1)

    cost = hybrid_cost(
        semantic_cost,
        topology_cost
    )

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

    return transport, cost
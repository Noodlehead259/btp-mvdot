import torch


def knn_graph(x, k=10, chunk_size=512):
    n = x.size(0)
    edges = []

    x_norm = (x * x).sum(dim=1)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        chunk = x[start:end]

        distance = (
            x_norm[start:end].unsqueeze(1)
            + x_norm.unsqueeze(0)
            - 2.0 * (chunk @ x.t())
        )

        distance = torch.clamp(distance, min=0.0)

        local_k = min(k + 1, n)

        _, indices = torch.topk(
            distance,
            k=local_k,
            dim=1,
            largest=False
        )

        indices = indices[:, 1:]

        rows = torch.arange(
            start,
            end,
            device=x.device
        ).unsqueeze(1).expand_as(indices)

        edges.append(
            torch.stack(
                [
                    rows.reshape(-1),
                    indices.reshape(-1)
                ],
                dim=0
            )
        )

    edges = torch.cat(edges, dim=1)

    reverse_edges = edges.flip(0)

    edges = torch.cat(
        [edges, reverse_edges],
        dim=1
    )

    values = torch.ones(
        edges.size(1),
        device=x.device,
        dtype=x.dtype
    )

    graph = torch.sparse_coo_tensor(
        edges,
        values,
        size=(n, n),
        device=x.device
    ).coalesce()

    graph = torch.sparse_coo_tensor(
        graph.indices(),
        torch.ones(
            graph.indices().size(1),
            device=x.device,
            dtype=x.dtype
        ),
        size=(n, n),
        device=x.device
    ).coalesce()

    return graph


def graph_laplacian(graph):
    indices = graph.indices()
    values = graph.values()

    n = graph.size(0)

    degree = torch.zeros(
        n,
        device=graph.device,
        dtype=values.dtype
    )

    degree.scatter_add_(
        0,
        indices[0],
        values
    )

    return degree


def manifold_regularization(z, graph, degree):
    indices = graph.indices()
    values = graph.values()

    source = indices[0]
    target = indices[1]

    z_squared = (z * z).sum(dim=1)

    first_term = torch.sum(
        degree * z_squared
    )

    dot_products = (
        z[source] * z[target]
    ).sum(dim=1)

    second_term = torch.sum(
        values * dot_products
    )

    return first_term - second_term


def manifold_matching_loss(
    transport,
    cost,
    z,
    graph,
    degree,
    lambda_graph=0.05
):
    transport_cost = torch.sum(
        transport * cost
    )

    graph_cost = manifold_regularization(
        z,
        graph,
        degree
    )

    loss = (
        transport_cost
        + lambda_graph * graph_cost
    )

    return loss, transport_cost, graph_cost
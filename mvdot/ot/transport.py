import torch
import torch.nn.functional as f

from mvdot.ot.sinkhorn import sinkhorn


def sample_to_cluster_cost(features, centers):
    features = f.normalize(features, p=2, dim=1)
    centers = f.normalize(centers, p=2, dim=1)

    similarity = features @ centers.t()

    cost = 2.0 - 2.0 * similarity

    return cost


def sample_to_cluster_transport(
    features,
    centers,
    weights,
    epsilon=0.1,
    iterations=100
):
    n = features.size(0)

    cost = sample_to_cluster_cost(
        features,
        centers
    )

    source_mass = torch.ones(
        n,
        device=features.device,
        dtype=features.dtype
    ) / n

    transport = sinkhorn(
        cost,
        source_mass,
        weights,
        epsilon=epsilon,
        iterations=iterations
    )

    return transport, cost
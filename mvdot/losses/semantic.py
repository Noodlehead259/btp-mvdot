import torch


def cluster_probabilities(
    features,
    centers,
    temperature=1.0
):
    similarity = features @ centers.t()

    probabilities = torch.softmax(
        similarity / temperature,
        dim=1
    )

    return probabilities


def semantic_compatibility(
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

    return compatibility


def semantic_cost(
    probabilities1,
    probabilities2
):
    compatibility = semantic_compatibility(
        probabilities1,
        probabilities2
    )

    return -torch.log(
        compatibility
    )


def semantic_matching_loss(
    transport,
    probabilities1,
    probabilities2
):
    compatibility = semantic_compatibility(
        probabilities1,
        probabilities2
    )

    return torch.sum(
        transport
        * torch.log(compatibility)
    )


def view_weights(
    transport_costs,
    eps=1e-8
):
    distances = torch.stack(
        transport_costs
    )

    inverse_distances = 1.0 / (
        distances + eps
    )

    weights = (
        inverse_distances
        / inverse_distances.sum()
    )

    return weights


def total_semantic_matching_loss(
    transports,
    probabilities,
    transport_costs
):
    weights = view_weights(
        transport_costs
    )

    num_views = len(
        probabilities
    )

    loss = torch.tensor(
        0.0,
        device=probabilities[0].device,
        dtype=probabilities[0].dtype
    )

    for m in range(num_views):
        for n in range(num_views):
            if m == n:
                continue

            pair_loss = semantic_matching_loss(
                transports[(m, n)],
                probabilities[m],
                probabilities[n]
            )

            loss = (
                loss
                + weights[m]
                * weights[n]
                * pair_loss
            )

    return loss, weights
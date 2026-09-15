import torch
import torch.nn.functional as f


def spherical_kmeans(
    x,
    num_clusters=10,
    iterations=20,
    seed=42
):
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)

    x = f.normalize(
        x,
        p=2,
        dim=1
    )

    indices = torch.randperm(
        x.size(0),
        generator=generator,
        device=x.device
    )[:num_clusters]

    centers = x[indices].clone()

    for _ in range(iterations):
        similarity = x @ centers.t()

        assignments = similarity.argmax(
            dim=1
        )

        new_centers = torch.zeros_like(
            centers
        )

        for k in range(num_clusters):
            mask = assignments == k

            if mask.any():
                new_centers[k] = x[mask].mean(
                    dim=0
                )
            else:
                index = torch.randint(
                    0,
                    x.size(0),
                    (1,),
                    generator=generator,
                    device=x.device
                )

                new_centers[k] = x[
                    index
                ].squeeze(0)

        new_centers = f.normalize(
            new_centers,
            p=2,
            dim=1
        )

        if torch.allclose(
            centers,
            new_centers,
            atol=1e-5
        ):
            centers = new_centers
            break

        centers = new_centers

    return centers


def initialize_barycenter(
    z1,
    z2,
    num_clusters=10,
    iterations=20,
    seed=42
):
    x = torch.cat(
        [z1, z2],
        dim=0
    )

    centers = spherical_kmeans(
        x,
        num_clusters=num_clusters,
        iterations=iterations,
        seed=seed
    )

    weights = torch.ones(
        num_clusters,
        device=x.device,
        dtype=x.dtype
    ) / num_clusters

    return centers, weights


def update_barycenter(
    features,
    transports,
    centers,
    weights,
    alpha=0.98
):
    num_clusters = centers.size(0)

    new_centers = torch.zeros_like(
        centers
    )

    cluster_mass = torch.zeros(
        num_clusters,
        device=centers.device,
        dtype=centers.dtype
    )

    for z, transport in zip(
        features,
        transports
    ):
        mass = transport.sum(
            dim=0
        )

        cluster_mass += mass

        new_centers += (
            transport.t() @ z
        )

    new_centers = (
        new_centers
        / cluster_mass.clamp_min(1e-12).unsqueeze(1)
    )

    new_centers = f.normalize(
        new_centers,
        p=2,
        dim=1
    )

    target_weights = (
        cluster_mass
        / cluster_mass.sum().clamp_min(1e-12)
    )

    new_weights = (
        alpha * weights
        + (1.0 - alpha) * target_weights
    )

    new_weights = (
        new_weights
        / new_weights.sum().clamp_min(1e-12)
    )

    return new_centers, new_weights
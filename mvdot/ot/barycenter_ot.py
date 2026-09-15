import torch
import torch.nn.functional as f


def spherical_kmeans(
    x,
    num_clusters=10,
    iterations=20,
    seed=42
):
    x = f.normalize(
        x,
        p=2,
        dim=1
    )

    generator = torch.Generator(
        device=x.device
    )

    generator.manual_seed(
        seed
    )

    indices = torch.randperm(
        x.size(0),
        generator=generator,
        device=x.device
    )[:num_clusters]

    centers = x[
        indices
    ].clone()

    for _ in range(
        iterations
    ):
        similarity = (
            x @ centers.t()
        )

        assignments = similarity.argmax(
            dim=1
        )

        new_centers = torch.zeros_like(
            centers
        )

        for k in range(
            num_clusters
        ):
            mask = (
                assignments == k
            )

            if mask.any():
                new_centers[k] = x[
                    mask
                ].mean(
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
    features,
    num_clusters=10,
    iterations=20,
    seed=42
):
    centers = spherical_kmeans(
        features,
        num_clusters=num_clusters,
        iterations=iterations,
        seed=seed
    )

    weights = torch.ones(
        num_clusters,
        device=features.device,
        dtype=features.dtype
    ) / num_clusters

    return (
        centers,
        weights
    )
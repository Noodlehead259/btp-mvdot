import os
import sys

import torch
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score
)

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from mvdot.data.preprocessing import (
    load_images,
    load_labels,
    create_noisy_view
)

from mvdot.data.dataset import noisy_mnist
from mvdot.data.dataloader import create_dataloader

from mvdot.models.autoencoder import autoencoder
from mvdot.models.matching import matching_network
from mvdot.models.barycenter import barycenter

from mvdot.models.transport_matching import (
    cross_view_affinity,
    build_bipartite_graph,
    bfs_topology,
    topology_cost,
    semantic_cost_from_probabilities,
    cross_view_transport
)

from mvdot.losses.semantic import (
    cluster_probabilities
)


def clustering_accuracy(
    labels,
    predictions,
    num_clusters
):
    confusion = torch.zeros(
        num_clusters,
        num_clusters,
        dtype=torch.long
    )

    for true_label, predicted_label in zip(
        labels,
        predictions
    ):
        confusion[
            predicted_label,
            true_label
        ] += 1

    dp = torch.full(
        (1 << num_clusters,),
        -1,
        dtype=torch.long
    )

    dp[0] = 0

    for mask in range(
        1 << num_clusters
    ):
        used = mask.bit_count()

        if used >= num_clusters:
            continue

        current = dp[
            mask
        ]

        if current < 0:
            continue

        for label in range(
            num_clusters
        ):
            if mask & (
                1 << label
            ):
                continue

            new_mask = (
                mask
                | (
                    1 << label
                )
            )

            value = (
                current
                + confusion[
                    used,
                    label
                ]
            )

            if value > dp[
                new_mask
            ]:
                dp[
                    new_mask
                ] = value

    best = dp[
        (1 << num_clusters) - 1
    ].item()

    return (
        best
        / len(labels)
    )


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


checkpoint = torch.load(
    "checkpoints/mvdot_final.pt",
    map_location=device
)


input_dim = checkpoint[
    "input_dim"
]

hidden_dim = checkpoint[
    "hidden_dim"
]

latent_dim = checkpoint[
    "latent_dim"
]

num_clusters = checkpoint[
    "num_clusters"
]

aligned_rate = checkpoint[
    "aligned_rate"
]

sigma = checkpoint[
    "sigma"
]

epsilon = checkpoint[
    "epsilon"
]

top_k = checkpoint[
    "cross_view_top_k"
]

sinkhorn_iterations = checkpoint[
    "sinkhorn_iterations"
]

batch_size = checkpoint[
    "batch_size"
]


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

matching = matching_network(
    feature_dim=latent_dim
).to(device)

bary = barycenter(
    num_clusters=num_clusters,
    feature_dim=latent_dim
).to(device)


model1.load_state_dict(
    checkpoint["model1"]
)

model2.load_state_dict(
    checkpoint["model2"]
)

matching.load_state_dict(
    checkpoint["matching"]
)

bary.load_state_dict(
    checkpoint["barycenter"]
)


model1.eval()
model2.eval()
matching.eval()
bary.eval()


centers, weights = bary()


images = load_images(
    "data/train-images.idx3-ubyte"
)

labels = load_labels(
    "data/train-labels.idx1-ubyte"
)

noisy_images = create_noisy_view(
    images,
    sigma=sigma,
    seed=checkpoint["seed"]
)


dataset = noisy_mnist(
    images,
    noisy_images,
    labels,
    aligned_rate=aligned_rate,
    seed=checkpoint["seed"]
)


loader = create_dataloader(
    dataset.view1,
    dataset.view2,
    dataset.labels,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False
)


reference_cost1 = 0.0
reference_cost2 = 0.0
number_of_batches = 0


print(
    "device:",
    device
)

print(
    "samples:",
    len(dataset)
)

print(
    "aligned rate:",
    aligned_rate
)

print(
    "sigma:",
    sigma
)


print()
print(
    "selecting reference view..."
)


with torch.no_grad():

    for view1, view2, _ in loader:

        view1 = view1.to(
            device,
            non_blocking=True
        )

        view2 = view2.to(
            device,
            non_blocking=True
        )

        z1 = model1.encode(
            view1
        )

        z2 = model2.encode(
            view2
        )

        h1 = matching(
            z1
        )

        h2 = matching(
            z2
        )

        t1, cost1 = (
            __import__(
                "mvdot.ot.transport",
                fromlist=[
                    "sample_to_cluster_transport"
                ]
            ).sample_to_cluster_transport(
                h1,
                centers,
                weights,
                epsilon=epsilon,
                iterations=sinkhorn_iterations
            )
        )

        t2, cost2 = (
            __import__(
                "mvdot.ot.transport",
                fromlist=[
                    "sample_to_cluster_transport"
                ]
            ).sample_to_cluster_transport(
                h2,
                centers,
                weights,
                epsilon=epsilon,
                iterations=sinkhorn_iterations
            )
        )

        reference_cost1 += (
            t1 * cost1
        ).sum().item()

        reference_cost2 += (
            t2 * cost2
        ).sum().item()

        number_of_batches += 1


reference_cost1 /= number_of_batches
reference_cost2 /= number_of_batches


if reference_cost1 <= reference_cost2:
    reference_view = 1
else:
    reference_view = 2


inverse1 = 1.0 / (
    reference_cost1 + 1e-8
)

inverse2 = 1.0 / (
    reference_cost2 + 1e-8
)


view_weight1 = (
    inverse1
    / (
        inverse1
        + inverse2
    )
)

view_weight2 = (
    inverse2
    / (
        inverse1
        + inverse2
    )
)


print(
    "view 1 barycenter cost:",
    reference_cost1
)

print(
    "view 2 barycenter cost:",
    reference_cost2
)

print(
    "reference view:",
    reference_view
)

print(
    "view weights:",
    view_weight1,
    view_weight2
)


predictions = []
true_labels = []


print()
print(
    "performing consensus clustering..."
)


with torch.no_grad():

    for batch_index, (
        view1,
        view2,
        batch_labels
    ) in enumerate(loader):

        view1 = view1.to(
            device,
            non_blocking=True
        )

        view2 = view2.to(
            device,
            non_blocking=True
        )

        z1 = model1.encode(
            view1
        )

        z2 = model2.encode(
            view2
        )

        h1 = matching(
            z1
        )

        h2 = matching(
            z2
        )

        probabilities1 = cluster_probabilities(
            h1,
            centers
        )

        probabilities2 = cluster_probabilities(
            h2,
            centers
        )

        edges, _ = cross_view_affinity(
            h1,
            h2,
            centers,
            top_k=top_k
        )

        graph_edges = build_bipartite_graph(
            edges,
            h1.size(0),
            h2.size(0)
        )

        topology = bfs_topology(
            graph_edges,
            h1.size(0),
            h2.size(0)
        ).to(device)

        topology_matrix = topology_cost(
            topology
        )

        semantic_matrix = (
            semantic_cost_from_probabilities(
                probabilities1,
                probabilities2
            )
        )

        p12, _ = cross_view_transport(
            semantic_matrix,
            topology_matrix,
            epsilon=epsilon,
            iterations=sinkhorn_iterations
        )

        row_mass = p12.sum(
            dim=1,
            keepdim=True
        )

        aligned_h2 = (
            p12 @ h2
        ) / (
            row_mass + 1e-8
        )

        aligned_h2 = torch.nn.functional.normalize(
            aligned_h2,
            p=2,
            dim=1
        )

        col_mass = p12.sum(
            dim=0,
            keepdim=True
        ).t()

        aligned_h1 = (
            p12.t() @ h1
        ) / (
            col_mass + 1e-8
        )

        aligned_h1 = torch.nn.functional.normalize(
            aligned_h1,
            p=2,
            dim=1
        )

        if reference_view == 1:
            fused = (
                view_weight1 * h1
                + view_weight2 * aligned_h2
            )
        else:
            fused = (
                view_weight1 * aligned_h1
                + view_weight2 * h2
            )

        fused = torch.nn.functional.normalize(
            fused,
            p=2,
            dim=1
        )

        similarity = (
            fused @ centers.t()
        )

        batch_predictions = similarity.argmax(
            dim=1
        )

        predictions.append(
            batch_predictions.cpu()
        )

        true_labels.append(
            batch_labels
        )

        if (
            batch_index == 0
            or (batch_index + 1) % 25 == 0
            or batch_index + 1 == len(loader)
        ):
            print(
                "batch:",
                batch_index + 1,
                "/",
                len(loader)
            )


predictions = torch.cat(
    predictions
).numpy()

true_labels = torch.cat(
    true_labels
).numpy()


acc = clustering_accuracy(
    torch.tensor(true_labels),
    torch.tensor(predictions),
    num_clusters
)

nmi = normalized_mutual_info_score(
    true_labels,
    predictions
)

ari = adjusted_rand_score(
    true_labels,
    predictions
)


print()
print(
    "final results"
)

print(
    "samples:",
    len(true_labels)
)

print(
    "reference view:",
    reference_view
)

print(
    "view weight 1:",
    view_weight1
)

print(
    "view weight 2:",
    view_weight2
)

print()
print(
    "acc:",
    acc
)

print(
    "nmi:",
    nmi
)

print(
    "ari:",
    ari
)
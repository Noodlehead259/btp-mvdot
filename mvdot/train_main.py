import os
import sys

import torch
import torch.nn.functional as f

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

from mvdot.losses.reconstruction import reconstruction_loss
from mvdot.losses.manifold import (
    knn_graph,
    graph_laplacian,
    manifold_matching_loss
)

from mvdot.losses.semantic import (
    cluster_probabilities,
    total_semantic_matching_loss
)

from mvdot.ot.barycenter_ot import (
    initialize_barycenter
)

from mvdot.ot.transport import (
    sample_to_cluster_transport
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

input_dim = 784
hidden_dim = 512
latent_dim = 128
num_clusters = 10

batch_size = 256

warmup_epochs = 10
train_epochs = 50

aligned_rate = 0.5
sigma = 0.25

learning_rate = 3e-4

epsilon = 0.1

lambda_graph = 0.05

alpha = 0.98

knn_k = 10
cross_view_top_k = 5

sinkhorn_iterations = 100

seed = 42


torch.manual_seed(
    seed
)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        seed
    )


print(
    "device:",
    device
)

print(
    "loading data..."
)


images = load_images(
    "data/train-images.idx3-ubyte"
)

labels = load_labels(
    "data/train-labels.idx1-ubyte"
)

noisy_images = create_noisy_view(
    images,
    sigma=sigma,
    seed=seed
)


dataset = noisy_mnist(
    images,
    noisy_images,
    labels,
    aligned_rate=aligned_rate,
    seed=seed
)


loader = create_dataloader(
    dataset.view1,
    dataset.view2,
    dataset.labels,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)


print(
    "samples:",
    len(dataset)
)

print(
    "batches:",
    len(loader)
)

print(
    "aligned rate:",
    aligned_rate
)

print(
    "noise sigma:",
    sigma
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

matching = matching_network(
    feature_dim=latent_dim
).to(device)

bary = barycenter(
    num_clusters=num_clusters,
    feature_dim=latent_dim
).to(device)


criterion = reconstruction_loss()


warmup_optimizer = torch.optim.Adam(
    list(model1.parameters())
    + list(model2.parameters()),
    lr=learning_rate
)


print()
print(
    "starting autoencoder warm-up..."
)


model1.train()
model2.train()

for epoch in range(
    warmup_epochs
):
    epoch_loss = 0.0

    for view1, view2, _ in loader:

        view1 = view1.to(
            device,
            non_blocking=True
        )

        view2 = view2.to(
            device,
            non_blocking=True
        )

        warmup_optimizer.zero_grad(
            set_to_none=True
        )

        z1, xhat1 = model1(
            view1
        )

        z2, xhat2 = model2(
            view2
        )

        loss1 = criterion(
            view1,
            xhat1
        )

        loss2 = criterion(
            view2,
            xhat2
        )

        loss = (
            loss1
            + loss2
        )

        loss.backward()

        warmup_optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(
        loader
    )

    print(
        "warmup epoch:",
        epoch + 1,
        "/",
        warmup_epochs,
        "loss:",
        epoch_loss
    )


print()
print(
    "autoencoder warm-up complete"
)


print(
    "extracting warm-up representations..."
)


model1.eval()
model2.eval()
matching.eval()


all_h1 = []
all_h2 = []


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

        all_h1.append(
            h1
        )

        all_h2.append(
            h2
        )


h1_all = torch.cat(
    all_h1,
    dim=0
)

h2_all = torch.cat(
    all_h2,
    dim=0
)


initial_features = torch.cat(
    [
        h1_all,
        h2_all
    ],
    dim=0
)


print(
    "initializing barycenter..."
)


initial_centers, initial_weights = (
    initialize_barycenter(
        initial_features,
        num_clusters=num_clusters,
        iterations=20,
        seed=seed
    )
)


bary.set_centers(
    initial_centers
)

bary.weights.copy_(
    initial_weights
)


del h1_all
del h2_all
del initial_features
del all_h1
del all_h2


if torch.cuda.is_available():
    torch.cuda.empty_cache()


print(
    "centers shape:",
    bary.centers.shape
)

print(
    "weights:",
    bary.weights
)


optimizer = torch.optim.Adam(
    list(model1.parameters())
    + list(model2.parameters())
    + list(matching.parameters())
    + list(bary.parameters()),
    lr=learning_rate
)


print()
print(
    "starting mvdot training..."
)


model1.train()
model2.train()
matching.train()
bary.train()


for epoch in range(
    train_epochs
):

    epoch_loss = 0.0
    epoch_lrec = 0.0
    epoch_lmm = 0.0
    epoch_lsm = 0.0

    for batch_index, (
        view1,
        view2,
        _
    ) in enumerate(loader):

        view1 = view1.to(
            device,
            non_blocking=True
        )

        view2 = view2.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        z1, xhat1 = model1(
            view1
        )

        z2, xhat2 = model2(
            view2
        )

        h1 = matching(
            z1
        )

        h2 = matching(
            z2
        )

        centers, weights = bary()

        lrec1 = criterion(
            view1,
            xhat1
        )

        lrec2 = criterion(
            view2,
            xhat2
        )

        lrec = (
            lrec1
            + lrec2
        )


        t1, cost1 = (
            sample_to_cluster_transport(
                h1,
                centers,
                weights,
                epsilon=epsilon,
                iterations=sinkhorn_iterations
            )
        )

        t2, cost2 = (
            sample_to_cluster_transport(
                h2,
                centers,
                weights,
                epsilon=epsilon,
                iterations=sinkhorn_iterations
            )
        )


        graph1 = knn_graph(
            z1,
            k=knn_k,
            chunk_size=64
        )

        graph2 = knn_graph(
            z2,
            k=knn_k,
            chunk_size=64
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
            degree1,
            lambda_graph=lambda_graph
        )

        manifold2 = manifold_matching_loss(
            t2,
            cost2,
            z2,
            graph2,
            degree2,
            lambda_graph=lambda_graph
        )


        transport_cost1 = manifold1[1]
        transport_cost2 = manifold2[1]

        graph_cost1 = manifold1[2]
        graph_cost2 = manifold2[2]


        lmm = (
            transport_cost1
            + transport_cost2
            + lambda_graph * (
                graph_cost1
                + graph_cost2
            )
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
            sigma=1.0,
            top_k=cross_view_top_k
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
        ).to(
            device
        )


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


        if not torch.isfinite(
            loss
        ):
            print(
                "non-finite loss at batch:",
                batch_index + 1
            )

            raise RuntimeError(
                "non-finite loss encountered"
            )


        loss.backward()

        optimizer.step()


        with torch.no_grad():

            bary.normalize_centers()

            bary.update_weights(
                [
                    t1.detach(),
                    t2.detach()
                ],
                momentum=alpha
            )


        epoch_loss += loss.item()
        epoch_lrec += lrec.item()
        epoch_lmm += lmm.item()
        epoch_lsm += lsm.item()


        if (
            batch_index == 0
            or (batch_index + 1) % 25 == 0
            or batch_index + 1 == len(loader)
        ):
            print(
                "epoch:",
                epoch + 1,
                "/",
                train_epochs,
                "batch:",
                batch_index + 1,
                "/",
                len(loader),
                "loss:",
                loss.item()
            )


    epoch_loss /= len(
        loader
    )

    epoch_lrec /= len(
        loader
    )

    epoch_lmm /= len(
        loader
    )

    epoch_lsm /= len(
        loader
    )


    print()
    print(
        "epoch:",
        epoch + 1,
        "average loss:",
        epoch_loss
    )

    print(
        "lrec:",
        epoch_lrec
    )

    print(
        "lmm:",
        epoch_lmm
    )

    print(
        "lsm:",
        epoch_lsm
    )

    print(
        "view weights:",
        view_weights.detach()
    )

    print(
        "barycenter weight sum:",
        bary.weights.sum().item()
    )

    print()


os.makedirs(
    "checkpoints",
    exist_ok=True
)


checkpoint = {
    "model1": model1.state_dict(),
    "model2": model2.state_dict(),
    "matching": matching.state_dict(),
    "barycenter": bary.state_dict(),
    "input_dim": input_dim,
    "hidden_dim": hidden_dim,
    "latent_dim": latent_dim,
    "num_clusters": num_clusters,
    "aligned_rate": aligned_rate,
    "sigma": sigma,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "epsilon": epsilon,
    "lambda_graph": lambda_graph,
    "alpha": alpha,
    "knn_k": knn_k,
    "cross_view_top_k": cross_view_top_k,
    "sinkhorn_iterations": sinkhorn_iterations,
    "warmup_epochs": warmup_epochs,
    "train_epochs": train_epochs,
    "seed": seed
}


checkpoint_path = (
    "checkpoints/mvdot_final.pt"
)


torch.save(
    checkpoint,
    checkpoint_path
)


print()
print(
    "checkpoint saved"
)

print(
    "path:",
    checkpoint_path
)

print()
print(
    "mvdot training complete"
)
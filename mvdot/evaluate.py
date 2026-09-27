import os
import sys

import torch
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score
)

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from mvdot.data.preprocessing import (
    create_noisy_view,
    load_mnist
)
from mvdot.data.dataset import noisy_mnist
from mvdot.models.autoencoder import autoencoder
from mvdot.models.barycenter import barycenter
from mvdot.models.matching import matching_network
from mvdot.inference.consensus import (
    compute_cross_view_plan,
    fuse_batch_representations,
    predict_clusters,
    select_reference_view
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

        current = dp[mask]

        if current < 0:
            continue

        for label in range(num_clusters):
            if mask & (1 << label):
                continue

            new_mask = mask | (1 << label)
            value = current + confusion[used, label]

            if value > dp[new_mask]:
                dp[new_mask] = value

    best = dp[(1 << num_clusters) - 1].item()

    return best / len(labels)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

checkpoint = torch.load(
    "checkpoints/mvdot_final.pt",
    map_location=device
)

model1 = autoencoder(
    input_dim=checkpoint["input_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    latent_dim=checkpoint["latent_dim"]
).to(device)

model2 = autoencoder(
    input_dim=checkpoint["input_dim"],
    hidden_dim=checkpoint["hidden_dim"],
    latent_dim=checkpoint["latent_dim"]
).to(device)

matching = matching_network(
    feature_dim=checkpoint["latent_dim"]
).to(device)

bary = barycenter(
    num_clusters=checkpoint["num_clusters"],
    feature_dim=checkpoint["latent_dim"]
).to(device)

model1.load_state_dict(checkpoint["model1"])
model2.load_state_dict(checkpoint["model2"])
matching.load_state_dict(checkpoint["matching"])
bary.load_state_dict(checkpoint["barycenter"])

model1.eval()
model2.eval()
matching.eval()
bary.eval()

centers, weights = bary()

images, labels = load_mnist("data")

noisy_images = create_noisy_view(
    images,
    sigma=checkpoint["sigma"],
    seed=checkpoint["seed"]
)

dataset = noisy_mnist(
    images,
    noisy_images,
    labels,
    aligned_rate=checkpoint["aligned_rate"],
    seed=checkpoint["seed"]
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=checkpoint["batch_size"],
    shuffle=False,
    drop_last=False,
    pin_memory=True
)

(
    reference_view,
    view_weight1,
    view_weight2,
    reference_cost1,
    reference_cost2
) = select_reference_view(
    loader,
    model1,
    model2,
    matching,
    centers,
    weights,
    device,
    checkpoint["epsilon"],
    checkpoint["sinkhorn_iterations"]
)

predictions = []
true_labels = []
offset = 0

with torch.no_grad():
    for view1, view2, batch_labels in loader:
        batch_size = batch_labels.size(0)
        end = offset + batch_size

        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)

        h1 = matching(model1.encode(view1))
        h2 = matching(model2.encode(view2))

        p12 = compute_cross_view_plan(
            h1,
            h2,
            centers,
            checkpoint["epsilon"],
            checkpoint["sinkhorn_iterations"],
            checkpoint["cross_view_top_k"]
        )

        fused = fuse_batch_representations(
            h1,
            h2,
            dataset.aligned_mask[offset:end],
            reference_view,
            view_weight1,
            view_weight2,
            p12
        )

        predictions.append(
            predict_clusters(fused, centers).cpu()
        )
        true_labels.append(batch_labels)
        offset = end

predictions = torch.cat(predictions).numpy()
true_labels = torch.cat(true_labels).numpy()

acc = clustering_accuracy(
    torch.tensor(true_labels),
    torch.tensor(predictions),
    checkpoint["num_clusters"]
)

nmi = normalized_mutual_info_score(
    true_labels,
    predictions
)

ari = adjusted_rand_score(
    true_labels,
    predictions
)

print("samples:", len(true_labels))
print("aligned samples:", int(dataset.aligned_mask.sum()))
print("reference view:", reference_view)
print("view 1 barycenter cost:", reference_cost1)
print("view 2 barycenter cost:", reference_cost2)
print("view weight 1:", view_weight1)
print("view weight 2:", view_weight2)
print("acc:", acc)
print("nmi:", nmi)
print("ari:", ari)

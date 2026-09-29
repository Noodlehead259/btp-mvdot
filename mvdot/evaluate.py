import os
import sys
from collections.abc import Mapping

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
from mvdot.losses.confidence import (
    cross_view_agreement,
    normalize_source_mass,
    sample_confidence
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


def load_autoencoder_checkpoint(
    model,
    state_dict: Mapping
):
    """Load current or pre-gate checkpoints without hiding unrelated errors."""
    model_keys = set(model.state_dict())
    checkpoint_keys = set(state_dict)
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    gate_keys = {
        "reliability_gate.weight",
        "reliability_gate.bias"
    }

    if unexpected_keys or not missing_keys.issubset(gate_keys):
        raise RuntimeError(
            "Incompatible autoencoder checkpoint: "
            f"missing={sorted(missing_keys)}, "
            f"unexpected={sorted(unexpected_keys)}"
        )

    model.load_state_dict(
        state_dict,
        strict=False
    )

    if missing_keys:
        with torch.no_grad():
            model.reliability_gate.weight.zero_()
            model.reliability_gate.bias.fill_(10.0)
        print(
            "Warning: legacy checkpoint detected; "
            "reliability gates were not stored. "
            "Using an open gate for evaluation."
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

load_autoencoder_checkpoint(
    model1,
    checkpoint["model1"]
)
load_autoencoder_checkpoint(
    model2,
    checkpoint["model2"]
)
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
confidence_values1 = []
confidence_values2 = []
gate_values1 = []
gate_values2 = []
offset = 0

with torch.no_grad():
    for view1, view2, batch_labels in loader:
        batch_size = batch_labels.size(0)
        end = offset + batch_size

        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)

        raw_z1, z1, gate1, xhat1 = model1(view1)
        raw_z2, z2, gate2, xhat2 = model2(view2)

        h1 = matching(z1)
        h2 = matching(z2)

        probabilities1 = torch.softmax(
            h1 @ centers.t(),
            dim=1
        )

        probabilities2 = torch.softmax(
            h2 @ centers.t(),
            dim=1
        )

        p12 = compute_cross_view_plan(
            h1,
            h2,
            centers,
            checkpoint["epsilon"],
            checkpoint["sinkhorn_iterations"],
            checkpoint["cross_view_top_k"]
        )

        agreement1, agreement2 = cross_view_agreement(
            probabilities1,
            probabilities2,
            p12
        )

        confidence1 = sample_confidence(
            (view1, xhat1),
            probabilities1,
            agreement1,
            reconstruction_temperature=checkpoint.get(
                "confidence_reconstruction_temperature",
                0.05
            ),
            reconstruction_exponent=checkpoint.get(
                "confidence_reconstruction_exponent",
                1.0
            ),
            posterior_exponent=checkpoint.get(
                "confidence_posterior_exponent",
                1.0
            ),
            agreement_exponent=checkpoint.get(
                "confidence_agreement_exponent",
                1.0
            )
        )

        confidence2 = sample_confidence(
            (view2, xhat2),
            probabilities2,
            agreement2,
            reconstruction_temperature=checkpoint.get(
                "confidence_reconstruction_temperature",
                0.05
            ),
            reconstruction_exponent=checkpoint.get(
                "confidence_reconstruction_exponent",
                1.0
            ),
            posterior_exponent=checkpoint.get(
                "confidence_posterior_exponent",
                1.0
            ),
            agreement_exponent=checkpoint.get(
                "confidence_agreement_exponent",
                1.0
            )
        )

        source_mass1 = normalize_source_mass(
            confidence1
        )

        source_mass2 = normalize_source_mass(
            confidence2
        )

        p12 = compute_cross_view_plan(
            h1,
            h2,
            centers,
            checkpoint["epsilon"],
            checkpoint["sinkhorn_iterations"],
            checkpoint["cross_view_top_k"],
            source_mass=source_mass1,
            target_mass=source_mass2
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
        confidence_values1.append(confidence1.cpu())
        confidence_values2.append(confidence2.cpu())
        gate_values1.append(gate1.mean(dim=1).cpu())
        gate_values2.append(gate2.mean(dim=1).cpu())
        offset = end

predictions = torch.cat(predictions).numpy()
true_labels = torch.cat(true_labels).numpy()
confidence_values1 = torch.cat(
    confidence_values1
).numpy()
confidence_values2 = torch.cat(
    confidence_values2
).numpy()
gate_values1 = torch.cat(
    gate_values1
).numpy()
gate_values2 = torch.cat(
    gate_values2
).numpy()

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
print(
    "mean confidence view 1:",
    confidence_values1.mean()
)
print(
    "mean confidence view 2:",
    confidence_values2.mean()
)
print(
    "mean gate activation view 1:",
    gate_values1.mean()
)
print(
    "mean gate activation view 2:",
    gate_values2.mean()
)

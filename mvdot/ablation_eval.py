"""Evaluation-only ablation harness for the reliability-gated MvDOT model.

The script never writes checkpoints and does not alter the training entrypoint.
It evaluates the supplied checkpoint under controlled gate interventions and
data perturbations, emitting one JSON object per condition and a flat CSV.
"""

import argparse
import csv
import json
import os
import sys
from copy import deepcopy

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvdot.data.dataset import noisy_mnist
from mvdot.data.preprocessing import create_noisy_view, load_mnist
from mvdot.inference.consensus import (
    compute_cross_view_plan,
    fuse_batch_representations,
    predict_clusters,
    select_reference_view,
)
from mvdot.models.autoencoder import autoencoder
from mvdot.models.barycenter import barycenter
from mvdot.models.matching import matching_network


def clustering_accuracy(labels, predictions, num_clusters):
    confusion = torch.zeros(num_clusters, num_clusters, dtype=torch.long)
    for true_label, predicted_label in zip(labels, predictions):
        confusion[predicted_label, true_label] += 1
    dp = torch.full((1 << num_clusters,), -1, dtype=torch.long)
    dp[0] = 0
    for mask in range(1 << num_clusters):
        used = mask.bit_count()
        if used >= num_clusters or dp[mask] < 0:
            continue
        for label in range(num_clusters):
            if not mask & (1 << label):
                new_mask = mask | (1 << label)
                dp[new_mask] = max(dp[new_mask], dp[mask] + confusion[used, label])
    return float(dp[-1].item() / len(labels))


def load_autoencoder_checkpoint(model, state_dict):
    model_keys, checkpoint_keys = set(model.state_dict()), set(state_dict)
    missing, unexpected = model_keys - checkpoint_keys, checkpoint_keys - model_keys
    gate_keys = {"reliability_gate.weight", "reliability_gate.bias"}
    if unexpected or not missing.issubset(gate_keys):
        raise RuntimeError(f"incompatible checkpoint: missing={missing}, unexpected={unexpected}")
    model.load_state_dict(state_dict, strict=False)
    if missing:
        with torch.no_grad():
            model.reliability_gate.weight.zero_()
            model.reliability_gate.bias.fill_(10.0)


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/mvdot_final.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-prefix", default="ablation_results")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--noise-levels", default="0,0.125,0.25,0.375,0.5")
    p.add_argument("--alignment-rates", default="1,0.75,0.5,0.25,0")
    p.add_argument("--gate-interventions", default="gated,raw,open,closed,inverted,shuffled")
    p.add_argument("--gate-loss-lambdas", default="0,0.01,0.1")
    p.add_argument("--ablation-epochs", type=int, default=1)
    p.add_argument("--skip-gate-loss-ablation", action="store_true")
    p.add_argument("--device", default=None, choices=["cpu", "cuda"])
    return p.parse_args()


def _load_models(checkpoint, device):
    dims = [checkpoint.get(k, v) for k, v in
            (("input_dim", 784), ("hidden_dim", 512), ("latent_dim", 128))]
    models = [autoencoder(*dims).to(device) for _ in range(2)]
    for model, state in zip(models, (checkpoint["model1"], checkpoint["model2"])):
        load_autoencoder_checkpoint(model, state)
        model.eval()
    matching = matching_network(feature_dim=dims[2]).to(device)
    matching.load_state_dict(checkpoint["matching"])
    matching.eval()
    bary = barycenter(checkpoint.get("num_clusters", 10), dims[2]).to(device)
    bary.load_state_dict(checkpoint["barycenter"])
    bary.eval()
    return models, matching, bary


def _parse_floats(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _gate_override(raw_gate, mode, generator):
    if mode == "gated":
        return raw_gate
    if mode == "raw":
        return torch.ones_like(raw_gate)
    if mode == "open":
        return torch.ones_like(raw_gate)
    if mode == "closed":
        return torch.zeros_like(raw_gate)
    if mode == "inverted":
        return 1.0 - raw_gate
    if mode == "shuffled":
        return raw_gate[torch.randperm(raw_gate.size(0), generator=generator,
                                       device=raw_gate.device)]
    raise ValueError("unknown gate intervention: " + mode)


def evaluate_condition(models, matching, bary, dataset, checkpoint, device,
                       mode="gated", seed=42):
    batch_size = checkpoint.get("batch_size", 256)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, drop_last=False)
    centers, weights = bary()
    epsilon = checkpoint.get("epsilon", 0.1)
    iterations = checkpoint.get("sinkhorn_iterations", 100)
    top_k = checkpoint.get("cross_view_top_k", 5)
    ref = select_reference_view(loader, models[0], models[1], matching, centers,
                                weights, device, epsilon, iterations)
    rng = torch.Generator(device=device).manual_seed(seed)
    predictions, labels, gates, suppressions, rec_errors, correct = [], [], [], [], [], []
    offset = 0
    with torch.no_grad():
        for view1, view2, batch_labels in loader:
            view1, view2 = view1.to(device), view2.to(device)
            raws, zs, gs, xhats, hs = [], [], [], [], []
            for model, view in zip(models, (view1, view2)):
                raw, gated, gate, _ = model(view)
                intervention = _gate_override(gate, mode, rng)
                z = intervention * raw
                xhat = model.decode(z)
                raws.append(raw); zs.append(z); gs.append(intervention)
                xhats.append(xhat); hs.append(matching(z))
            p12 = compute_cross_view_plan(hs[0], hs[1], centers, epsilon,
                                          iterations, top_k)
            n = len(batch_labels)
            fused = fuse_batch_representations(
                hs[0], hs[1], dataset.aligned_mask[offset:offset + n],
                ref[0], ref[1], ref[2], p12)
            pred = predict_clusters(fused, centers)
            predictions.append(pred.cpu()); labels.append(batch_labels)
            gates.append(torch.cat([gs[0].mean(1), gs[1].mean(1)]).cpu().reshape(2, -1).mean(0))
            raw_norm = torch.cat([raws[0].norm(dim=1), raws[1].norm(dim=1)]).reshape(2, -1).mean(0)
            gated_norm = torch.cat([zs[0].norm(dim=1), zs[1].norm(dim=1)]).reshape(2, -1).mean(0)
            suppressions.append((1 - gated_norm / raw_norm.clamp_min(1e-8)).cpu())
            err = ((view1 - xhats[0]).pow(2).mean(1) +
                   (view2 - xhats[1]).pow(2).mean(1)) / 2
            rec_errors.append(err.cpu())
            correct.append((pred.cpu() == batch_labels).float())
            offset += n
    pred, label = torch.cat(predictions).numpy(), torch.cat(labels).numpy()
    gate, suppression = torch.cat(gates).numpy(), torch.cat(suppressions).numpy()
    rec_error, is_correct = torch.cat(rec_errors).numpy(), torch.cat(correct).numpy()
    return {
        "samples": int(len(label)), "aligned_samples": int(dataset.aligned_mask.sum()),
        "reference_view": int(ref[0]), "acc": clustering_accuracy(torch.from_numpy(label),
        torch.from_numpy(pred), checkpoint.get("num_clusters", 10)),
        "nmi": float(normalized_mutual_info_score(label, pred)),
        "ari": float(adjusted_rand_score(label, pred)),
        "mean_gate": float(gate.mean()), "gate_std": float(gate.std()),
        "gate_correct_corr": _pearson(gate, is_correct),
        "gate_reconstruction_error_corr": _pearson(gate, rec_error),
        "mean_feature_suppression": float(suppression.mean()),
        "suppression_std": float(suppression.std()),
        "suppressed_fraction": float((suppression > 0.5).mean()),
    }


def _write_results(rows, prefix):
    with open(prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    keys = sorted({k for row in rows for k in row})
    with open(prefix + ".csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = _args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    max_samples = args.max_samples or checkpoint.get("num_training_samples")
    images, labels = load_mnist(args.data_dir)
    if max_samples:
        images, labels = images[:max_samples], labels[:max_samples]
    models, matching, bary = _load_models(checkpoint, device)
    sigma = checkpoint.get("sigma", 0.25)
    aligned = checkpoint.get("aligned_rate", 0.5)
    rows = []

    def run(test, condition, view2, rate=aligned, mode="gated", model_set=None):
        ds = noisy_mnist(images, view2, labels, aligned_rate=rate,
                         seed=checkpoint.get("seed", 42))
        result = evaluate_condition(*(model_set or (models, matching, bary)), ds,
                                    checkpoint, device, mode)
        rows.append({"test": test, "condition": condition, "sigma": sigma,
                     "alignment_rate": rate, **result})
        print(json.dumps(rows[-1], sort_keys=True))

    base_view = create_noisy_view(images, sigma=sigma, seed=checkpoint.get("seed", 42))
    for mode in args.gate_interventions.split(","):
        run("controlled_baseline_vs_gate" if mode in ("gated", "raw") else
            "gate_interventions", mode, base_view, mode=mode)
    for level in _parse_floats(args.noise_levels):
        run("noise_sensitivity", str(level), create_noisy_view(
            images, sigma=level, seed=checkpoint.get("seed", 42)))
    for rate in _parse_floats(args.alignment_rates):
        run("alignment_sensitivity", str(rate), base_view, rate=rate)
    if not args.skip_gate_loss_ablation:
        for lam in _parse_floats(args.gate_loss_lambdas):
            cloned = deepcopy(models)
            for model in cloned: model.train()
            optimizer = torch.optim.Adam([p for m in cloned for p in m.parameters()],
                                         lr=3e-4)
            ds = noisy_mnist(images, base_view, labels, aligned_rate=aligned,
                             seed=checkpoint.get("seed", 42))
            loader = torch.utils.data.DataLoader(ds, batch_size=checkpoint.get("batch_size", 256),
                                                 shuffle=False)
            for _ in range(args.ablation_epochs):
                for v1, v2, _ in loader:
                    optimizer.zero_grad()
                    loss = 0
                    for model, view in zip(cloned, (v1.to(device), v2.to(device))):
                        raw, gated, gate, xhat = model(view)
                        loss = loss + (view - xhat).pow(2).mean() + lam * (gate.mean() - checkpoint.get("gate_target", 0.7)).pow(2)
                    loss.backward(); optimizer.step()
            for model in cloned: model.eval()
            run("gate_loss_ablation", str(lam), base_view, model_set=(cloned, matching, bary))
    _write_results(rows, args.output_prefix)
    print(f"wrote {len(rows)} rows to {args.output_prefix}.json and .csv")


if __name__ == "__main__":
    main()

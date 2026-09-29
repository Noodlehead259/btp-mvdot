import torch
import torch.nn.functional as f

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

from mvdot.ot.transport import (
    sample_to_cluster_transport
)


def select_reference_view(
    loader,
    model1,
    model2,
    matching,
    centers,
    weights,
    device,
    epsilon,
    sinkhorn_iterations
):
    reference_cost1 = 0.0
    reference_cost2 = 0.0
    number_of_batches = 0

    model1.eval()
    model2.eval()
    matching.eval()

    with torch.no_grad():
        for batch in loader:
            view1, view2 = batch[0], batch[1]

            view1 = view1.to(
                device,
                non_blocking=True
            )

            view2 = view2.to(
                device,
                non_blocking=True
            )

            h1 = matching(
                model1.encode(view1)
            )

            h2 = matching(
                model2.encode(view2)
            )

            t1, cost1 = sample_to_cluster_transport(
                h1,
                centers,
                weights,
                epsilon=epsilon,
                iterations=sinkhorn_iterations
            )

            t2, cost2 = sample_to_cluster_transport(
                h2,
                centers,
                weights,
                epsilon=epsilon,
                iterations=sinkhorn_iterations
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

    return (
        reference_view,
        view_weight1,
        view_weight2,
        reference_cost1,
        reference_cost2
    )


def compute_cross_view_plan(
    h1,
    h2,
    centers,
    epsilon,
    sinkhorn_iterations,
    top_k,
    source_mass=None,
    target_mass=None
):
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
    ).to(
        h1.device
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
        source_mass=source_mass,
        target_mass=target_mass,
        epsilon=epsilon,
        iterations=sinkhorn_iterations
    )

    return p12


def fuse_batch_representations(
    h1,
    h2,
    is_aligned,
    reference_view,
    view_weight1,
    view_weight2,
    p12
):
    row_mass = p12.sum(
        dim=1,
        keepdim=True
    ).clamp_min(
        1e-8
    )

    realigned_h2 = (
        p12 @ h2
    ) / row_mass

    realigned_h2 = f.normalize(
        realigned_h2,
        p=2,
        dim=1
    )

    col_mass = p12.sum(
        dim=0,
        keepdim=True
    ).t().clamp_min(
        1e-8
    )

    realigned_h1 = (
        p12.t() @ h1
    ) / col_mass

    realigned_h1 = f.normalize(
        realigned_h1,
        p=2,
        dim=1
    )

    is_aligned = is_aligned.to(
        h1.device
    ).bool()

    if reference_view == 1:
        fused_aligned = (
            view_weight1 * h1
            + view_weight2 * h2
        )

        fused_unaligned = (
            view_weight1 * h1
            + view_weight2 * realigned_h2
        )
    else:
        fused_aligned = (
            view_weight1 * h1
            + view_weight2 * h2
        )

        fused_unaligned = (
            view_weight1 * realigned_h1
            + view_weight2 * h2
        )

    fused = torch.where(
        is_aligned.unsqueeze(1),
        fused_aligned,
        fused_unaligned
    )

    return f.normalize(
        fused,
        p=2,
        dim=1
    )


def predict_clusters(
    fused,
    centers
):
    similarity = (
        fused @ centers.t()
    )

    return similarity.argmax(
        dim=1
    )

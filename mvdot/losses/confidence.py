import math

import torch


def reconstruction_confidence(
    inputs,
    reconstructions,
    temperature=1.0
):
    error = (
        inputs - reconstructions
    ).pow(2).mean(
        dim=tuple(range(1, inputs.dim()))
    )

    return torch.exp(
        -error / max(temperature, 1e-8)
    )


def posterior_confidence(
    probabilities
):
    entropy = -torch.sum(
        probabilities
        * torch.log(
            probabilities.clamp_min(1e-12)
        ),
        dim=1
    )

    maximum_entropy = math.log(
        probabilities.size(1)
    )

    return (
        1.0
        - entropy / max(maximum_entropy, 1e-8)
    ).clamp(
        min=0.0,
        max=1.0
    )


def cross_view_agreement(
    probabilities1,
    probabilities2,
    transport,
    eps=1e-8
):
    compatibility = (
        probabilities1
        @ probabilities2.t()
    )

    row_mass = transport.sum(
        dim=1
    ).clamp_min(eps)

    agreement1 = (
        transport * compatibility
    ).sum(
        dim=1
    ) / row_mass

    column_mass = transport.sum(
        dim=0
    ).clamp_min(eps)

    agreement2 = (
        transport * compatibility
    ).sum(
        dim=0
    ) / column_mass

    return (
        agreement1.clamp(0.0, 1.0),
        agreement2.clamp(0.0, 1.0)
    )


def sample_confidence(
    reconstruction,
    posterior,
    agreement,
    reconstruction_temperature=0.05,
    reconstruction_exponent=1.0,
    posterior_exponent=1.0,
    agreement_exponent=1.0,
    eps=1e-8
):
    reconstruction_score = reconstruction_confidence(
        reconstruction[0],
        reconstruction[1],
        temperature=reconstruction_temperature
    )

    posterior_score = posterior_confidence(
        posterior
    )

    confidence = (
        reconstruction_score.clamp_min(eps)
        .pow(reconstruction_exponent)
        * posterior_score.clamp_min(eps)
        .pow(posterior_exponent)
        * agreement.clamp_min(eps)
        .pow(agreement_exponent)
    )

    return confidence.clamp_min(eps)


def normalize_source_mass(
    confidence,
    eps=1e-8
):
    confidence = confidence.clamp_min(eps)
    return confidence / confidence.sum().clamp_min(eps)

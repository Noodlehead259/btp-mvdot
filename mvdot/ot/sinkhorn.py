import torch


def sinkhorn(
    cost,
    source_mass,
    target_mass,
    epsilon=0.1,
    iterations=100
):
    log_kernel = -cost / epsilon

    log_source = torch.log(source_mass)
    log_target = torch.log(target_mass)

    log_u = torch.zeros_like(source_mass)
    log_v = torch.zeros_like(target_mass)

    for _ in range(iterations):
        log_u = log_source - torch.logsumexp(
            log_kernel + log_v.unsqueeze(0),
            dim=1
        )

        log_v = log_target - torch.logsumexp(
            log_kernel + log_u.unsqueeze(1),
            dim=0
        )

    transport = torch.exp(
        log_u.unsqueeze(1)
        + log_kernel
        + log_v.unsqueeze(0)
    )

    return transport
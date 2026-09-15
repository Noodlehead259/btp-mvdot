import torch
import torch.nn as nn
import torch.nn.functional as f


class barycenter(nn.Module):
    def __init__(
        self,
        num_clusters=10,
        feature_dim=128
    ):
        super().__init__()

        self.num_clusters = num_clusters
        self.feature_dim = feature_dim

        self.centers = nn.Parameter(
            torch.randn(
                num_clusters,
                feature_dim
            )
        )

        self.register_buffer(
            "weights",
            torch.ones(
                num_clusters
            ) / num_clusters
        )

        self.normalize_centers()

    def forward(self):
        return (
            f.normalize(
                self.centers,
                p=2,
                dim=1
            ),
            self.weights
        )

    @torch.no_grad()
    def set_centers(self, centers):
        self.centers.copy_(
            f.normalize(
                centers,
                p=2,
                dim=1
            )
        )

    @torch.no_grad()
    def normalize_centers(self):
        self.centers.copy_(
            f.normalize(
                self.centers,
                p=2,
                dim=1
            )
        )

    @torch.no_grad()
    def update_weights(
        self,
        transports,
        momentum=0.98
    ):
        average_mass = torch.stack(
            [
                transport.sum(
                    dim=0
                )
                for transport in transports
            ]
        ).mean(
            dim=0
        )

        average_mass = (
            average_mass
            / average_mass.sum().clamp_min(
                1e-12
            )
        )

        self.weights.mul_(
            momentum
        ).add_(
            average_mass,
            alpha=1.0 - momentum
        )

        self.weights.div_(
            self.weights.sum().clamp_min(
                1e-12
            )
        )
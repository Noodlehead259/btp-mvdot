import torch
import torch.nn as nn
import torch.nn.functional as f


class barycenter(nn.Module):
    def __init__(self, num_clusters=10, feature_dim=128):
        super().__init__()

        self.num_clusters = num_clusters
        self.feature_dim = feature_dim

        self.centers = nn.Parameter(
            torch.randn(num_clusters, feature_dim)
        )

        self.register_buffer(
            "weights",
            torch.ones(num_clusters) / num_clusters
        )

        self.normalize_centers()

    def normalize_centers(self):
        with torch.no_grad():
            self.centers.copy_(
                f.normalize(self.centers, p=2, dim=1)
            )

    def forward(self):
        return self.centers, self.weights
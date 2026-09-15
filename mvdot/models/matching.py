import torch.nn as nn
import torch.nn.functional as f


class matching_network(nn.Module):
    def __init__(
        self,
        feature_dim=128
    ):
        super().__init__()

        self.fc = nn.Linear(
            feature_dim,
            feature_dim
        )

    def forward(self, z):
        return f.normalize(
            self.fc(z),
            p=2,
            dim=1
        )
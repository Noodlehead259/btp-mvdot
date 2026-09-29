import torch
import torch.nn as nn


class autoencoder(nn.Module):
    def __init__(
        self,
        input_dim=784,
        hidden_dim=512,
        latent_dim=128
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                latent_dim
            )
        )

        self.reliability_gate = nn.Linear(
            latent_dim,
            latent_dim
        )

        nn.init.zeros_(
            self.reliability_gate.weight
        )
        nn.init.constant_(
            self.reliability_gate.bias,
            2.0
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                latent_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                input_dim
            ),
            nn.Sigmoid()
        )

    def encode(self, x):
        _, gated_z, _, _ = self.forward(x)
        return gated_z

    def encode_with_gate(self, x):
        z = self.encoder(x)
        gate = torch.sigmoid(
            self.reliability_gate(z)
        )
        gated_z = gate * z
        return z, gated_z, gate

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, gated_z, gate = self.encode_with_gate(
            x
        )
        x_hat = self.decode(gated_z)

        return z, gated_z, gate, x_hat
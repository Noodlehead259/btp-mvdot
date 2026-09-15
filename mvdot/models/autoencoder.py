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
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)

        return z, x_hat
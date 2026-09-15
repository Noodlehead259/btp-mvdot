import sys
import os
import torch

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )
)

from mvdot.models.autoencoder import autoencoder


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = autoencoder(
    input_dim=784,
    hidden_dim=512,
    latent_dim=128
).to(device)

x = torch.randn(
    32,
    784,
    device=device
)

z, x_hat = model(x)

print("device:", device)
print("input shape:", x.shape)
print("latent shape:", z.shape)
print("output shape:", x_hat.shape)
print(
    "parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)
import torch.nn as nn
import torch.nn.functional as f


class reconstruction_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_hat):
        return f.mse_loss(x_hat, x)
import torch
import torch.nn as nn


class Shallow(nn.Module):
    """
    Shallow feed forward network.
    """
    def __init__(self, in_dim, out_dim, width):

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=width, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=width, out_features=out_dim, bias=False),
        )

    def forward(self, x):

        return self.layers(x)

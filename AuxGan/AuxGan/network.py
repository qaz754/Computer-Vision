
import torch
import torch.nn as nn

from util import Flatten

class discriminator(nn.Module):
    def __init__(self, num_classes):
        super(discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
        )

        self.src = nn.Sequential(
            nn.Linear(256, 1),
        )

        self.cls = nn.Sequential(
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return self.src(x), self.cls(x)


def generator(noise_dim=96):
    """
    Build and return a PyTorch model implementing the architecture above.
    """
    model = nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 784),
        nn.Tanh()
    )
    return model
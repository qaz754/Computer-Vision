
import torch
import torch.nn as nn

from util import Flatten, Unflatten

def discriminator(batch_size):
    """
    From https://arxiv.org/abs/1511.06434.pdf
    """
    model = nn.Sequential(
        Unflatten(batch_size, 1, 28, 28),
        nn.Conv2d(1, 32, 4, 2, 1),
        nn.InstanceNorm2d(32),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(32, 64, 4, 2, 1),
        nn.InstanceNorm2d(64),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(64, 1, 4, 1, 1)
    )

    return model

def generator(batch_size, noise_dim=96):
    """
    From https://arxiv.org/pdf/1606.03657.pdf
    """
    model = nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 7*7*128),
        nn.ReLU(),
        nn.BatchNorm1d(7*7*128),
        Unflatten(batch_size, 128, 7, 7),
        nn.ConvTranspose2d(128, 64, 4, stride=2,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
        nn.Tanh(),
        Flatten()
    )
    return model
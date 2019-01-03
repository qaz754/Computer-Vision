
import torch
import torch.nn as nn

from util import Flatten, Unflatten

def discriminator(batch_size, n_classes = 10):
    """
    From https://arxiv.org/abs/1511.06434.pdf
    """
    model = nn.Sequential(
        Unflatten(batch_size, 1, 28, 28),
        nn.Conv2d(1, 32, 5, stride=1),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.4),
        nn.Conv2d(32, 64, 5, stride=1),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.4),
        Flatten(),
        nn.Linear(4 * 4 * 64, 4 * 4 * 64),
        nn.Dropout(0.4),
        nn.LeakyReLU(0.01),
        nn.Linear(4 * 4 * 64, n_classes)
    )

    return model

class discriminator(nn.Module):

    def __int__(self, batch_size, n_classes=10):
        super(discriminator, self).__int__()

        self.model = nn.Sequential(
            Unflatten(batch_size, 1, 28, 28),
            nn.Conv2d(1, 32, 5, stride=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),
            nn.Conv2d(32, 64, 5, stride=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),
            Flatten(),
            nn.Linear(4 * 4 * 64, 4 * 4 * 64),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.01),
            nn.Linear(4 * 4 * 64, n_classes)
        )

def generator(batch_size, noise_dim=97):
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
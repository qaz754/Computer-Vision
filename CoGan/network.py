
import torch
import torch.nn as nn
import numpy as np

from util import Flatten, Unflatten

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class generator(nn.Module):

    def __init__(self, noise_shape, image_shape):
        super(generator, self).__init__()

        self.image_shape = image_shape

        def block(in_shape, out_shape, norm=True):
            model = [nn.Linear(in_shape, out_shape)]
            if norm:
                model.append(nn.BatchNorm1d(out_shape))
            model.append(nn.LeakyReLU(0.2))
            return model

        self.model = nn.Sequential(
            *block(noise_shape, 128, False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
        )

        self.image_a = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, True),
            nn.Linear(2048, int(np.prod(image_shape))),
            nn.Tanh()
        )

        self.image_b = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, True),
            nn.Linear(2048, int(np.prod(image_shape))),
            nn.Tanh()
        )

    def forward(self, x):

        x = self.model(x)
        image_a = self.image_a(x)
        image_b = self.image_b(x)

        return image_a.view(-1, 3, 28, 28), image_b.view(-1,  3, 28, 28)

class discriminator(nn.Module):

    def __init__(self, batch_size):
        super(discriminator, self).__init__()

        self.discrim_A = nn.Sequential(
            nn.Linear(4 * 4 * 64, 1)
        )
        self.discrim_B = nn.Sequential(
            nn.Linear(4 * 4 * 64, 1)
        )
        self.discrim = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, stride=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(4 * 4 * 64, 4 * 4 * 64),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x_a, x_b):
        x_a = self.discrim(x_a)
        x_a = self.discrim_A(x_a)

        x_b = self.discrim(x_b)
        x_b = self.discrim_B(x_b)

        return x_a, x_b





import torch
import torch.nn as nn
import numpy as np

import util

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class generator(nn.Module):

    def __init__(self, noise_shape, image_shape):
        super(generator, self).__init__()

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
            nn.Linear(1024, int(np.prod(image_shape))),
            nn.Tanh()
        )

    def forward(self, x):

        x = self.model(x)

        return x.view(-1, 1, 28, 28)

class discriminator(nn.Module):

    def __init__(self, image_shape):
        super(discriminator, self).__init__()

        self.discrim = nn.Sequential(
            util.Flatten(),
            nn.Linear(int(np.prod(image_shape)), 512),
            nn.LeakyReLU(0.20),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.20),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.20),
            nn.Linear(128, 1)
        )

    def forward(self, x):

        x = self.discrim(x)

        return x




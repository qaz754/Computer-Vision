
import torch
import torch.nn as nn

from util import Flatten, Unflatten


class discriminator(nn.Module):
    
    def __init__(self, batch_size, n_classes):
        super(discriminator, self).__init__()


        self.funnel = nn.Sequential(

            Unflatten(batch_size, 1, 28, 28),
            nn.Conv2d(1, 64, 4, stride=2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.BatchNorm2d(128),
            Flatten(),
            nn.Linear(3200, 1024),
            nn.LeakyReLU(0.01, inplace=True),
            #nn.BatchNorm1d(1024) #seems to hinder performance
        )

        self.discriminator = nn.Sequential(
            nn.Linear(1024, 1)
        )

        self.category = nn.Sequential(
            nn.Linear(1024, n_classes),
            nn.Softmax()
        )

        self.mu = nn.Sequential(
            nn.Linear(1024, 1)
        )

        self.var = nn.Sequential(
            nn.Linear(1024, 1)
        )



    def forward(self, x):

        x = self.funnel(x)

        #the outputs from the discriminator network
        d_output = self.discriminator(x)

        #the outputs from the recognition network
        #TODO handle Categorical and Continuous Variables
        #q_output = self.recognition(x)

        return d_output






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
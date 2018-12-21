
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AutoEncoder(nn.Module):

    def __init__(self, input_size, hidden_1, hidden_2, hidden_3, bottle_neck):
        super(AutoEncoder, self).__init__()

        #define the layers
        self.encode = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1), #batch_size * 16 * 10 * 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),                #batch_size * 16 * 5 * 5
            nn.Conv2d(16,8,3, stride=2, padding=2),   #batch_size * 8 * 3 * 3
            nn.ReLU(True),
        )

        self.mu_split = nn.Sequential(
            nn.MaxPool2d(2, stride=2)  # batch_size * 8 * 2 * 2
        )
        self.sigma_split = nn.Sequential(
            nn.MaxPool2d(2, stride=2)  # batch_size * 8 * 2 * 2
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # batch_size * 16 * 5 * 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # batch_size * 8 * 15 * 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # batch_size * 1 * 28 * 28
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.encode(x)


        #get mu and sigma

        mu = self.mu_split(x)
        sigma = self.sigma_split(x) #sigma^2

        '''
        mu, sigma = torch.chunk(self.mu_split(x), 2, dim=1)
        '''

        #randomly sample epsilon
        esp = (torch.randn(*mu.size()))

        #get the reparameterized function
        reparam = mu + sigma.mul(1/2).exp_() * Variable(esp.to(device), requires_grad=False)
        #reparam = mu + sigma.mul(1 / 2).exp_() * esp.to(device)

        x = self.decode(reparam)

        #reshape the output to B * 1 * 28 * 28
        return mu, sigma, x.view((-1, 1, 28, 28))


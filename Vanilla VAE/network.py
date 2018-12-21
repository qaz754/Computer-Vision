
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AutoEncoder(nn.Module):

    def __init__(self, input_size, hidden_1, hidden_2, hidden_3, bottle_neck):
        super(AutoEncoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_1, hidden_2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_2, hidden_3),
            nn.LeakyReLU(0.2),
        )
        self.mu_split = nn.Sequential(
            nn.Linear(hidden_3, bottle_neck)
        )
        self.sigma_split = nn.Sequential(
            nn.Linear(hidden_3, bottle_neck)
        )
        self.decode = nn.Sequential(
            nn.Linear(bottle_neck, hidden_3),
            nn.ReLU(True),
            nn.Linear(hidden_3, hidden_2),
            nn.ReLU(True),
            nn.Linear(hidden_2, hidden_1),
            nn.ReLU(True),
            nn.Linear(hidden_1, input_size),
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


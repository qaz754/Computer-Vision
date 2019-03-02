
import torch
import numpy as np
import torch.nn as nn

from util import Flatten


class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, opts, Activation='relu', Norm=False):
        super(Linear, self).__init__()

        steps = [nn.Linear(dim_in, dim_out)]

        if Norm:
            steps.append(nn.BatchNorm1d(dim_out))

        if Activation == 'relu':
            steps.append(nn.ReLU())
        elif Activation == 'lrelu':
            steps.append(nn.LeakyReLU(opts.lrelu_val))

        self.model = nn.Sequential(*steps)

    def forward(self, x):

        return self.model(x)

class discriminator(nn.Module):
    def __init__(self, opts):
        super(discriminator, self).__init__()

        steps = [Linear(opts.D_input_size, opts.D_hidden[0], opts, Activation=opts.D_activation)]

        if len(opts.D_hidden) > 1:
            for i in range(len(opts.D_hidden) - 1):
                steps.append(Linear(opts.D_hidden[i], opts.D_hidden[i + 1], opts, Activation=opts.D_activation))

        steps.append(Linear(opts.D_hidden[-1], opts.D_output_size, opts, Activation=''))

        self.model = nn.Sequential(*steps)

    def forward(self, x):
        return self.model(x)

class generator(nn.Module):
    def __init__(self, opts):
        super(generator, self).__init__()

        steps = [Linear(opts.noise_dim, opts.G_hidden[0], opts, Activation=opts.G_activation)]

        if len(opts.G_hidden) > 1:
            for i in range(len(opts.G_hidden) - 1):
                steps.append(Linear(opts.G_hidden[i], opts.G_hidden[i + 1], opts, Activation=opts.G_activation))

        steps.append(Linear(opts.G_hidden[-1], opts.G_output_size, opts, Activation=''))

        if opts.G_out_activation == 'tanh':
            final_activation = nn.Tanh()
        elif opts.G_out_activation == 'sigm':
            final_activation = nn.Sigmoid()

        steps.append(final_activation)

        self.model = nn.Sequential(*steps)

    def forward(self, x):
        return self.model(x)

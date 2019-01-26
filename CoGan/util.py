
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sample_noise(batch_size, dim):
    """
    Given the inputs batch_size, and dim return a tensor of values between (L, U)
    :param batch_size (int): size of the batch
    :param dim (int): size of the vector of dimensions
    :return: tensor of a random values between (L, U)

    Used to generate images in GANs
    """

    #TODO take the L and U as inputs of the function
    L = -1 #lower bound
    U = 1 #upper bound

    noise = (L - U) * torch.rand((batch_size, dim)) + U

    return noise

class Flatten(nn.Module):
    """
    Given a tensor of Batch * Color * Height * Width, flatten it and make it 1D.
    Used for Linear GANs

    Usable in nn.Sequential
    """
    def forward(self, x):

        B, C, H, W = x.size()

        return x.view(B, -1) #returns a vector that is B * (C * H * W)

class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (Batch, C*H*W) and reshapes it
    to produce an output of shape (Batch, C, H, W).

    C = Color Channels
    H = Heigh
    W = Width
    """

    def __init__(self, B=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()

        self.B = B
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.B, self.C, self.H, self.W)


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751

    :param input: PyTorch Tensor of shape (N, )
    :param target: PyTorch Tensor of shape (N, ). An indicator variable that is 0 or 1
    :return:
    """

    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()

    return loss.mean()


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss for Vanilla GANs

    :param logits_real: PyTorch Tensor of shape(N, ). Gives scores for the real data
    :param logits_fake: PyTorch Tensor of shape(N, ). Gives scores for the fake data
    :return: PyTorch Tensor containing the loss for the discriminator
    """

    labels = torch.ones(logits_real.size()).to(device) #label used to indicate whether it's real or not

    loss_real = nn.MSELoss()(logits_real, labels) #real data
    loss_fake = nn.MSELoss()(logits_fake, 1 - labels) #fake data

    loss = (loss_real + loss_fake) * 0.5

    return loss.to(device)

def generator_loss(logits_fake):
    """
    Computes the generator loss

    :param logits_fake: PyTorch Tensor of shape (N, ). Gives scores for the real data
    :return: PyTorch tensor containing the loss for the generator
    """

    labels = torch.ones(logits_fake.size()).to(device)

    loss = nn.MSELoss()(logits_fake, labels)

    return loss.to(device)


def LS_discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss for Vanilla GANs

    :param logits_real: PyTorch Tensor of shape(N, ). Gives scores for the real data
    :param logits_fake: PyTorch Tensor of shape(N, ). Gives scores for the fake data
    :return: PyTorch Tensor containing the loss for the discriminator
    """

    labels = torch.ones(logits_real.size()).to(device) #label used to indicate whether it's real or not

    #A, B, C values based on https://arxiv.org/pdf/1611.04076v3.pdf Page 8
    loss_real = 1/2 * ((logits_real - labels) ** 2).mean() #real data
    loss_fake = 1/2 * ((logits_fake) ** 2).mean() #fake data

    loss = loss_real + loss_fake

    return loss.to(device)

def LS_generator_loss(logits_fake):
    """
    Computes the generator loss

    :param logits_fake: PyTorch Tensor of shape (N, ). Gives scores for the real data
    :return: PyTorch tensor containing the loss for the generator
    """

    labels = torch.ones(logits_fake.size()).to(device)

    loss = 1/2 * ((logits_fake - labels) ** 2).mean()

    return loss.to(device)

def get_optimizer(model, lr=0.001):
    """
    Takes in PyTorch model and returns the Adam optimizer associated with it
    :param model:
    :param lr:
    :return:
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    return optimizer

def show_images(images, filename, iterations):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        '''global title'''
        plt.suptitle('LSGANs After %s iterations' %iterations)
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
        plt.savefig(filename)


def save_images_to_directory(image_tensor, directory, filename):
    directory = directory
    image = image_tensor.cpu().data

    save_name = os.path.join('%s' % directory, '%s' % filename)
    save_image(image, save_name)

    return filename

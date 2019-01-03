from collections import OrderedDict

import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.utils import save_image

import torch.nn.functional as F

#load mnist dataset and define network
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor()])

#Download and load the training data
batch_size = 128

trainset = datasets.MNIST('MNIST_data/', download=True, train= True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = datasets.MNIST('MNIST_data/', download=False, train=False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle=False)

from network import discriminator, generator
from util import get_optimizer, discriminator_loss, generator_loss
from train import run_vanilla_gan

NOISE_DIM = 96
N_CLASSES = 10

'''Discriminator'''
D = discriminator(batch_size, N_CLASSES).to(device)

'''Generator'''
G = generator(batch_size, NOISE_DIM + 1).to(device)

'''Optimizers'''
D_solver = get_optimizer(D)
G_solver = get_optimizer(G)

'''run training'''
run_vanilla_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, trainloader, num_classes = N_CLASSES, num_epochs=75)



'''Generate New Samples'''

from util import categorical_label_generator, sample_noise, show_images
from image_to_gif import image_to_gif
import os
import matplotlib.pyplot as plt

batch_size = 128
filelist = []

for i in range(0, 10):

    label_array = np.full((batch_size, 1), i)

    g_fake_seed = sample_noise(batch_size, NOISE_DIM).to(device)

    label_array = torch.from_numpy(label_array).long().to(device)

    g_fake_seed = torch.cat((g_fake_seed, label_array.float()), dim=1)

    fake_images = G(g_fake_seed).detach()

    imgs_numpy = fake_images.data.cpu().numpy()

    '''filename used for saving the image'''
    directory = './img/'
    filename = 'Generated_with_Label_%s.png' % i
    filelist.append(filename)

    filename = os.path.join('%s' % directory, '%s' % filename)

    show_images(imgs_numpy[0:16], filename, i)
    plt.show()
    print()

image_to_gif('./img/', filelist, duration=1, gifname='movie2')

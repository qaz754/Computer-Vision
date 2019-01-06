from collections import OrderedDict

import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.utils import save_image

from util import to_img


import torch.nn.functional as F

#load mnist dataset and define network
from torchvision import datasets, transforms

from network import AutoEncoder, discriminator
from trainer import trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor()])

#Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train= True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = datasets.MNIST('MNIST_data/', download=False, train=False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle=False)

lr = 0.0001
model = AutoEncoder(28 * 28, 512, 256, 64, 10).to(device)
discriminator = discriminator(10).to(device)

criterion = nn.BCELoss()

#updated during the reconstruction phase
reconstruction_parameters = list(model.parameters())

#updated during the regularization phase
discriminator_parameters = list(discriminator.parameters())
encoder_param = list(model.encode.parameters()) + list(model.mu_split.parameters()) + list(model.sigma_split.parameters())

#3 optimizers
reconstruction_optimizer = optim.Adam(reconstruction_parameters, lr=lr)
discriminator_optimizer = optim.Adam(discriminator_parameters, lr=lr)
encoder_optimizer = optim.Adam(encoder_param, lr=lr)

optim_list = [reconstruction_optimizer, discriminator_optimizer, encoder_optimizer]
model_list = [model, discriminator]
epochs = 100

trainer = trainer(epochs, trainloader, model_list, optim_list, criterion)

#trains the model
trainer.train()

idx = 0
for images, _ in iter(testloader):

    images = images.to(device)

    _, _, _, output = model(images.view((-1, 1, 28 * 28)))

    pic = output.cpu().data
    save_image(images.cpu().data, './img/test_real_{}.png'.format(idx))
    save_image(pic, './img/test_generated_{}.png'.format(idx))

    idx += 1




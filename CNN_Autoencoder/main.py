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

from network import AE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

#Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train= True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = datasets.MNIST('MNIST_data/', download=False, train=False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle=False)

import trainer

#Hyperparameters for our network
model = AE().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50


trainer = trainer.trainer(epochs, trainloader, model, optimizer, criterion)

#trains the model
trainer.train()


#test to see how the model works
images, labels = next(iter(testloader))


idx = 0
for images, _ in iter(testloader):

    images = images.to(device)

    output = model(images)

    pic = to_img(output.cpu().data)
    save_image(images.cpu().data, './img/test_real_{}.png'.format(idx))
    save_image(pic, './img/test_generated_{}.png'.format(idx))

    idx += 1





from collections import OrderedDict

import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

#load mnist dataset and define network
from torchvision import datasets, transforms

from network import Network


#Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

#Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train= True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST('MNIST_data/', download=False, train=False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle=False)

import trainer

#Hyperparameters for our network

output_size = 10
model = Network(3, output_size, 0.6)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

epochs = 15
print_every = 40
steps = 0

trainer = trainer.trainer(epochs, trainloader, model, optimizer, criterion, print_every)

#trains the model
trainer.train()



#test to see how the model works
images, labels = next(iter(trainloader))

#img = images[0].view(1, 784)
#turn off gradients to speed up this part

with torch.no_grad():
    logits = model.forward(images)


ps = F.softmax(logits, dim=1)

outputs = []

for images, labels in iter(testloader):

    #images = images.view(images.size()[0], -1)

    with torch.no_grad():
        output = model.forward(images)

    predictions = F.softmax(output, dim=1)

    outputs.append(predictions)

import util

y_label = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
img = images
util.pred_plotter(img, predictions, y_label)


def accuracy(net, loader):

    correct = 0.0
    total = 0.0

    for images, labels in iter(loader):

        output = net.forward(images)

        _, prediction = torch.max(output.data, 1)


        total += labels.shape[0] #accumulate by batch_size
        correct += (prediction == labels).sum() #accumulate by total_correct

    return correct, total

num_cor, num_total = accuracy(model, testloader)



images, labels = next(iter(trainloader))


output = model.forward(images)

_, prediction = torch.max(output.data, 1)


#Confusion matrix

confusion = torch.zeros(output_size, output_size)

for images, labels in iter(testloader):

    optimizer.zero_grad()

    output = model(images)
    _, prediction = torch.max(output.data, 1)

    confusion[prediction.item()][labels.item()] += 1

for i in range(output_size):

    confusion[i] = confusion[i] / confusion[i].sum()

util.confusion_plot(confusion.cpu(), list(y_label))


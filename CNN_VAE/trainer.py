
import util
import numpy as np

import torch

from torchvision.utils import save_image

from util import to_img

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch.nn import functional as F
class trainer():

    def __init__(self, epochs, trainloader, model, optimizer, criterion, print_every=50):

        self.epochs = epochs
        self.trainloader = trainloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.print_every = print_every

    def train(self):
        steps = 0

        for e in range(self.epochs):
            running_loss = 0

            for images, labels in iter(self.trainloader):
                steps += 1

                # flatten mnist images into a 784 long vector
                self.optimizer.zero_grad()

                # forward and backward passes

                images = images.to(device)

                mu, sigma, output = self.model.forward(images)

                LHS_loss = torch.sum(1 + sigma - mu ** 2 - sigma.exp()) / 2
                loss = F.binary_cross_entropy(output, images, reduction='sum') - LHS_loss
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if steps % self.print_every == 0:
                    print("Epoch: {}/{}...".format(e + 1, self.epochs),
                          "LossL {:.4f}".format(running_loss))

                running_loss = 0

            if e %1 == 0:
                pic = output.cpu().data
                save_image(pic, './img/image_{}.png'.format(e))

        torch.save(self.model.state_dict(), './CNN_VAE.pth')

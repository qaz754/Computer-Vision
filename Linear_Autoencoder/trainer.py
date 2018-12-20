
import util
import numpy as np

import torch

from torchvision.utils import save_image

from util import to_img

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

                output = self.model.forward(images.view((-1, 28 * 28)))
                loss = self.criterion(output, images)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if steps % self.print_every == 0:
                    print("Epoch: {}/{}...".format(e + 1, self.epochs),
                          "LossL {:.4f}".format(running_loss))

                running_loss = 0

            if e %1 == 0:
                pic = to_img(output.cpu().data)
                save_image(pic, './img/image_{}.png'.format(e))

        torch.save(self.model.state_dict(), './linear_autoencoder.pth')

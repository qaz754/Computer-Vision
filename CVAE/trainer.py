
import util
import numpy as np

import torch
import cv2_functions
import copy

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

            for images, _ in iter(self.trainloader):
                steps += 1

                # flatten mnist images into a 784 long vector
                self.optimizer.zero_grad()

                # forward and backward passes
                images_numpy = copy.deepcopy(images.numpy()) #deep copy to prevent creating a reference

                for i in range(images_numpy.shape[0]):
                    images_numpy[i] = cv2_functions.black_out(images_numpy[i], 2)
                    #save_image(torch.from_numpy(images_numpy[i]), './img/image_{}.png'.format(i))

                #util.show_images(images_numpy[0:16], 'test1', steps, 'Test %s')

                images_numpy = torch.from_numpy(images_numpy).float().to(device)

                images = images.to(device)

                mu, sigma, output = self.model.forward(images_numpy.view((-1, 28 * 28)))


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
                cropped_pic = images_numpy.cpu().data
                pic = output.cpu().data

                save_image(cropped_pic, './img/cropped_image_{}.png'.format(e))
                save_image(pic, './img/output_image_{}.png'.format(e))
                save_image(images, './img/original_image_{}.png'.format(e))

        torch.save(self.model.state_dict(), './linear_CVAE.pth')

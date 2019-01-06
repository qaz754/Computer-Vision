
import util
import os
import numpy as np
from image_to_gif import image_to_gif

import torch

from torchvision.utils import save_image

from util import to_img, sample_from_prior

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

        filelist = []

        for e in range(self.epochs):
            running_loss = 0

            for images, labels in iter(self.trainloader):
                steps += 1

                '''--------Reconstruction Phase--------'''
                self.optimizer[0].zero_grad()

                images = images.to(device)

                mu, sigma, encoder_sample, output = self.model[0].forward(images.view((-1, 28 * 28)))

                loss = F.binary_cross_entropy(output, images, reduction='sum')
                '''Unlike the regular VAE, this does not impose regularization on the Latent Vector'''
                '''Only has the reconstruction error comparing the input to the output'''
                loss.backward()
                self.optimizer[0].step()
                '''--------Reconstruction Phase--------'''


                '''Regularization Phase (Discriminator)'''

                self.optimizer[1].zero_grad()

                bottle_neck_size = encoder_sample.size()
                prior_sample = sample_from_prior(bottle_neck_size).to(device)


                prior_sample_logits = self.model[1](prior_sample)
                encoder_sample_logits = self.model[1](encoder_sample.detach())

                discrim_loss = util.discriminator_loss(prior_sample_logits, encoder_sample_logits)
                discrim_loss.backward()
                self.optimizer[1].step()
                '''Regularization Phase (Discriminator)'''


                '''Regularization Phase (Encoder)'''
                self.optimizer[2].zero_grad()
                _, _, encoder_sample, _ = self.model[0].forward(images.view((-1, 28 * 28)))
                encoder_sample_logits = self.model[1](encoder_sample)
                generator_loss = util.generator_loss(encoder_sample_logits)
                generator_loss.backward()
                self.optimizer[2].step()
                '''Regularization Phase (Encoder)'''


                running_loss += loss.item()

                if steps % self.print_every == 0:
                    print("Epoch: {}/{}...".format(e + 1, self.epochs),
                          "LossL {:.4f}".format(running_loss))

                running_loss = 0

            if e %1 == 0:
                pic = output.cpu().data
                directory = './img/'
                filename = 'image_%s.png' % e
                filelist.append(filename)

                filename = os.path.join('%s' % directory, '%s' % filename)
                save_image(pic, filename)


        torch.save(self.model[0].state_dict(), './linear_VAE.pth')
        torch.save(self.model[1].state_dict(), './Discriminator.pth')

        image_to_gif('./img/', filelist, duration=1)


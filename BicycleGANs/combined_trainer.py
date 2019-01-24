
import util
import os

from image_to_gif import image_to_gif

import torch
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from collections import deque
import numpy as np

class trainer():

    def __init__(self, epochs, trainloader, model, optimizer, criterion, print_every=5, lambd_l1=1, lamb_kl=1, lamb_latent=1, checkpoint_path = './model/checkpoint_424.pth', resume=False):

        self.epochs = epochs
        self.trainloader = trainloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.print_every = print_every
        self.lambd = lambd_l1
        self.kl_lamb = lamb_kl
        self.lamb_latent = lamb_latent

        self.checkpoint_path = checkpoint_path
        self.resume = resume

    def save_progress(self, epoch, loss):

        #TODO get rid of hardcoding and come up with an automated way.
        directory = './model/'
        filename = 'checkpoint_%s.pth' % epoch

        path = os.path.join('%s' % directory, '%s' % filename)

        torch.save({
            'epoch': epoch,
            'loss': loss,
            'model0_state_dict': self.model[0].state_dict(),
            'model1_state_dict': self.model[1].state_dict(),
            'model2_state_dict': self.model[2].state_dict(),
            'model3_state_dict': self.model[3].state_dict(),
            'optimizer0_state_dict': self.optimizer[0].state_dict(),
            'optimizer1_state_dict': self.optimizer[1].state_dict(),
            'optimizer2_state_dict': self.optimizer[2].state_dict(),
            'optimizer3_state_dict': self.optimizer[3].state_dict(),
        }, path)

        print("Saving Training Progress")

    def load_progress(self,):

        # TODO get rid of hardcoding and come up with an automated way.

        checkpoint = torch.load(self.checkpoint_path)

        self.model[0].load_state_dict(checkpoint['model0_state_dict'])
        self.model[1].load_state_dict(checkpoint['model1_state_dict'])
        self.model[2].load_state_dict(checkpoint['model2_state_dict'])
        self.model[3].load_state_dict(checkpoint['model3_state_dict'])

        self.optimizer[0].load_state_dict(checkpoint['optimizer0_state_dict'])
        self.optimizer[1].load_state_dict(checkpoint['optimizer1_state_dict'])
        self.optimizer[2].load_state_dict(checkpoint['optimizer2_state_dict'])
        self.optimizer[3].load_state_dict(checkpoint['optimizer3_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print("Training Loss From Last Session")
        return epoch, loss

    def train(self):
        steps = 0

        last_100_loss = deque(maxlen=100)

        #used to make gifs later
        pred_image_list = []
        target_image_list = []
        input_image_list = []

        train_loss = []

        directory = './img/'

        epoch = 0

        if self.resume:
            epoch, loss = self.load_progress()


        for e in range(self.epochs):
            running_loss = 0

            for input_image, target_image in iter(self.trainloader):
                steps += 1

                input_image = input_image.to(device)
                target_image = target_image.to(device)

                '''
                VAE
                '''

                '''Encoder Phase'''
                mu, sigma, Q_zb = self.model[3].forward(target_image)

                KL_loss = torch.sum(1 + sigma - mu ** 2 - sigma.exp()) * -0.5 * self.kl_lamb
                '''Encoder Phase'''

                image_pred = self.model[0](input_image, Q_zb)

                fake_logits = self.model[1](image_pred.detach())
                real_logits = self.model[1](target_image)

                cVAE_discrim_loss = util.discriminator_loss(real_logits, fake_logits)

                '''Regularization Phase (Discriminator)'''

                '''--------Reconstruction Phase--------'''
                cVAE_gen_logits = self.model[1](image_pred)
                cVAE_gen_loss = util.generator_loss(cVAE_gen_logits)

                image_pred = self.model[0](input_image, Q_zb)
                recon_loss = self.criterion(image_pred, target_image)




                '''Regularization Phase (Discriminator)'''

                #cLR-GAN discrim

                input_noise = torch.randn(input_image.shape[0], 8).to(device)

                image_pred = self.model[0](input_image, input_noise)

                fake_logits = self.model[2](image_pred.detach())
                real_logits = self.model[2](target_image)

                cLR_discrim_loss = util.discriminator_loss(real_logits, fake_logits)

                '''Regularization Phase (Discriminator)'''
                # cLR-GAN gen
                '''--------Reconstruction Phase--------'''

                gen_logits = self.model[2](image_pred)
                gen_loss = util.generator_loss(gen_logits)

                cLR_Gen_Loss = gen_loss

                '''--------Reconstruction Phase--------'''


                '''Update'''

                '''Generator Loss'''
                self.optimizer[3].zero_grad()
                self.optimizer[0].zero_grad()
                Generator_loss = cVAE_gen_loss + self.lambd * recon_loss + KL_loss + cLR_Gen_Loss
                Generator_loss.backward(retain_graph=True)
                self.optimizer[3].step()
                self.optimizer[0].step()
                '''Generator Loss'''

                '''cLR Encoder'''
                self.optimizer[0].zero_grad()
                _, _, Q_zb = self.model[3].forward(image_pred)
                latent_code_loss = self.lamb_latent * self.criterion(Q_zb, input_noise)
                latent_code_loss.backward()
                self.optimizer[0].step()
                '''cLR Encoder'''

                '''Discriminators cVAE'''
                self.optimizer[1].zero_grad()
                cVAE_discrim_loss.backward()
                self.optimizer[1].step()
                '''Discriminators cVAE'''

                '''Discriminators cLR'''
                self.optimizer[2].zero_grad()
                cLR_discrim_loss.backward()
                self.optimizer[2].step()
                '''Discriminators cLR'''

                running_loss += Generator_loss.item()
                train_loss.append(Generator_loss.item())
                last_100_loss.append(Generator_loss.item())

                if steps % self.print_every == 0:
                    print("Epoch: {}/{}...".format(e + 1, self.epochs),
                          "Last 100 {:.4f}".format(np.mean(last_100_loss)))

            if e % 1 == 0:

                pred_image_list.append(util.save_images_to_directory(image_pred, directory, 'g_pred_image_%s.png' % steps))
                input_image_list.append(
                    util.save_images_to_directory(input_image, directory, 'input_image_%s.png' % steps))
                target_image_list.append(
                    util.save_images_to_directory(target_image, directory, 'target_image_%s.png' % steps))

                util.raw_score_plotter(train_loss)
                if e % 10 == 0:
                    self.save_progress(e, np.mean(last_100_loss))

        self.save_progress(-1, np.mean(last_100_loss))

        image_to_gif('./img/', pred_image_list, duration=1, gifname='pred')
        image_to_gif('./img/', target_image_list, duration=1, gifname='target')
        image_to_gif('./img/', input_image_list, duration=1, gifname='input')


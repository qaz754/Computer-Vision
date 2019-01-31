import util
import os
import numpy as np
from image_to_gif import image_to_gif
from memory import ReplayBuffer
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from collections import deque

class trainer():

    def __init__(self, epochs, trainloader, model, optimizer, criterion, print_every=3000, lambda0=10, lambda1=0.01, lambda2=10, lambda3=0.01, lambda4=10, checkpoint_path = './model/checkpoint_0.pth', resume=False):

        self.epochs = epochs
        self.trainloader = trainloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.print_every = print_every

        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4

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
            'model4_state_dict': self.model[4].state_dict(),
            'model5_state_dict': self.model[5].state_dict(),
            'optimizer1_state_dict': self.optimizer[0].state_dict(),
            'optimizer2_state_dict': self.optimizer[1].state_dict(),
            'optimizer3_state_dict': self.optimizer[2].state_dict(),
        }, path)

        print("Saving Training Progress")

    def load_progress(self,):

        # TODO get rid of hardcoding and come up with an automated way.

        checkpoint = torch.load(self.checkpoint_path)

        self.model[0].load_state_dict(checkpoint['model0_state_dict'])
        self.model[1].load_state_dict(checkpoint['model1_state_dict'])
        self.model[2].load_state_dict(checkpoint['model2_state_dict'])
        self.model[3].load_state_dict(checkpoint['model3_state_dict'])
        self.model[4].load_state_dict(checkpoint['model4_state_dict'])
        self.model[5].load_state_dict(checkpoint['model5_state_dict'])

        self.optimizer[0].load_state_dict(checkpoint['optimizer1_state_dict'])
        self.optimizer[1].load_state_dict(checkpoint['optimizer2_state_dict'])
        self.optimizer[2].load_state_dict(checkpoint['optimizer3_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print("Training Loss From Last Session")
        return epoch, loss

    def train(self):
        steps = 0

        last_100_loss = deque(maxlen=100)
        # used to make gifs later
        fpred_image_list = []
        gpred_image_list = []
        target_image_list = []
        input_image_list = []

        train_loss = []

        directory = './img/'

        epoch = 0


        if self.resume:
            epoch, loss = self.load_progress()

        for e in range(self.epochs - epoch):

            running_loss = 0

            for input_image, target_image in iter(self.trainloader):

                steps += 1

                input_image = input_image.to(device)
                target_image = target_image.to(device)

                '''L_VAE'''
                mu1, z1 = self.model[0](input_image)
                mu2, z2 = self.model[1](target_image)

                '''within domain'''
                x1_recon = self.model[2](z1 + mu1)
                x2_recon = self.model[3](z2 + mu2)

                vae1_loss = self.lambda2 * self.criterion(x1_recon, input_image) + self.lambda1 * util.KL_Loss(mu1)
                vae2_loss = self.lambda2 * self.criterion(x2_recon, target_image) + self.lambda1 * util.KL_Loss(mu2)

                '''L_GAN'''
                '''cross domain'''
                x2_1_recon = self.model[2](z2 + mu2)
                x1_2_recon = self.model[3](z1 + mu1)

                x2_1_logit = self.model[4](x2_1_recon)
                x1_2_logit = self.model[5](x1_2_recon)

                Gen1_loss = self.lambda0 * util.generator_loss(x2_1_logit)
                Gen2_loss = self.lambda0 * util.generator_loss(x1_2_logit)

                x1_real_logit = self.model[4](input_image)
                x2_real_logit = self.model[5](target_image)

                D1_loss = self.lambda0 * util.discriminator_loss(x1_real_logit, self.model[4](x2_1_recon.detach()))
                D2_loss = self.lambda0 * util.discriminator_loss(x2_real_logit, self.model[5](x1_2_recon.detach()))

                '''L_Cycle Consistency'''
                mu2_1_cc, z2_1_cc = self.model[0](x2_1_recon)
                mu1_2_cc, z1_2_cc = self.model[1](x1_2_recon)

                x1_2_1_recon = self.model[2](z1_2_cc + mu1_2_cc)
                x2_1_2_recon = self.model[3](z2_1_cc + mu2_1_cc)

                cc1_loss = self.lambda4 * self.criterion(x1_2_1_recon, input_image) + self.lambda3 * util.KL_Loss(mu1_2_cc)
                cc2_loss = self.lambda4 * self.criterion(x2_1_2_recon, target_image) + self.lambda3 * util.KL_Loss(mu2_1_cc)

                gen_loss = Gen1_loss.item() + Gen2_loss.item()

                '''G Discriminator'''
                '''Model Update'''
                self.optimizer[0].zero_grad()
                total_gen_loss = vae1_loss + vae2_loss + Gen1_loss + Gen2_loss + cc1_loss + cc2_loss
                total_gen_loss.backward()
                self.optimizer[0].step()

                self.optimizer[1].zero_grad()
                D1_loss.backward()
                self.optimizer[1].step()

                '''G Discriminator'''
                self.optimizer[2].zero_grad()
                D2_loss.backward()
                self.optimizer[2].step()

                last_100_loss.append(gen_loss)
                running_loss += gen_loss
                train_loss.append(gen_loss)

                if steps % self.print_every == 0:

                    print('\rEpoch {}\tLoss: {:.4f}\n'.format(e, np.mean(last_100_loss)), end="")

                    fpred_image_list.append(util.save_images_to_directory(x1_2_recon, directory, 'x1_2_recon_image_%s.png' % steps))
                    gpred_image_list.append(util.save_images_to_directory(x2_1_recon, directory, 'x2_1_recon_image_%s.png' % steps))
                    input_image_list.append(util.save_images_to_directory(input_image, directory, 'input_image_%s.png' % steps))
                    target_image_list.append(util.save_images_to_directory(target_image, directory, 'target_image_%s.png' % steps))

                    util.raw_score_plotter(train_loss)

            self.save_progress(e, np.mean(last_100_loss))

        self.save_progress(-1, np.mean(last_100_loss))

        image_to_gif('./img/', fpred_image_list, duration=1, gifname='x1_2_recon_image')
        image_to_gif('./img/', gpred_image_list, duration=1, gifname='x2_1_recon_image')
        image_to_gif('./img/', target_image_list, duration=1, gifname='target')
        image_to_gif('./img/', input_image_list, duration=1, gifname='input')


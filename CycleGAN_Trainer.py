import util
import os
import numpy as np
from image_to_gif import image_to_gif
from memory import ReplayBuffer
import torch
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from collections import deque

class trainer():

    def __init__(self, epochs, trainloader, model, optimizer, criterion, print_every=15, cycle_lambda=1, checkpoint_path = './model/checkpoint_0.pth', resume=False):

        self.epochs = epochs
        self.trainloader = trainloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.print_every = print_every
        self.cycle_lambda = cycle_lambda

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
            'optimizer01_state_dict': self.optimizer[0].state_dict(),
            'optimizer2_state_dict' : self.optimizer[1].state_dict(),
            'optimizer3_state_dict' : self.optimizer[2].state_dict(),
        }, path)

        print("Saving Training Progress")

    def load_progress(self,):

        # TODO get rid of hardcoding and come up with an automated way.

        checkpoint = torch.load(self.checkpoint_path)

        self.model[0].load_state_dict(checkpoint['model0_state_dict'])
        self.model[1].load_state_dict(checkpoint['model1_state_dict'])
        self.model[2].load_state_dict(checkpoint['model2_state_dict'])
        self.model[3].load_state_dict(checkpoint['model3_state_dict'])

        self.optimizer[0].load_state_dict(checkpoint['optimizer01_state_dict'])
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

        F_deque = ReplayBuffer(50, 50, 0)
        G_deque = ReplayBuffer(50, 50, 0)

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

                '''Input(F) --> target(G)'''

                g_pred = self.model[0](input_image)
                G_deque.add(g_pred.detach())

                '''G_discriminator'''

                '''Train Discriminator'''

                g_fake_logits = self.model[2](G_deque.sample())
                g_real_logits = self.model[2](target_image)

                g_discrim_loss = util.discriminator_loss(g_real_logits, g_fake_logits)

                '''G_discriminator loss for the generator F'''

                F_gen_logit = self.model[2](g_pred)
                F_gen_loss = util.generator_loss(F_gen_logit)

                '''Input(G) --> target(F)'''

                f_pred = self.model[1](target_image)
                F_deque.add(f_pred.detach())

                '''F_discriminator'''

                f_fake_logits = self.model[3](F_deque.sample())
                f_real_logits = self.model[3](input_image)

                f_discrim_loss = util.discriminator_loss(f_real_logits, f_fake_logits)

                '''F_discriminator loss for the generator F'''
                G_gen_logits = self.model[3](f_pred)

                G_gen_loss = util.generator_loss(G_gen_logits)

                '''Cycle Consistency'''

                '''F(x) -> G(F(x) -> X'''
                FGF_pred = self.model[1](self.model[0](input_image))
                FGF_loss = self.criterion(FGF_pred, input_image)

                GFG_pred = self.model[0](self.model[1](target_image))
                GFG_loss = self.criterion(GFG_pred, target_image)

                cycle_loss = FGF_loss + GFG_loss

                '''Discriminators'''
                '''G Discriminator'''
                '''Model Update'''
                self.optimizer[0].zero_grad()
                gen_loss = F_gen_loss + G_gen_loss + self.cycle_lambda * cycle_loss
                gen_loss.backward()
                self.optimizer[0].step()

                self.optimizer[1].zero_grad()
                g_discrim_loss.backward()
                self.optimizer[1].step()

                '''G Discriminator'''
                self.optimizer[2].zero_grad()
                f_discrim_loss.backward()
                self.optimizer[2].step()

                last_100_loss.append(gen_loss.item())
                running_loss += gen_loss.item()
                train_loss.append(gen_loss.item())

                if steps % self.print_every == 0:

                    print('\rEpoch {}\tLoss: {:.4f}\n'.format(e, np.mean(last_100_loss)), end="")

                    fpred_image_list.append(util.save_images_to_directory(f_pred, directory, 'f_pred_image_%s.png' % steps))
                    gpred_image_list.append(util.save_images_to_directory(g_pred, directory, 'g_pred_image_%s.png' % steps))
                    input_image_list.append(util.save_images_to_directory(input_image, directory, 'input_image_%s.png' % steps))
                    target_image_list.append(util.save_images_to_directory(target_image, directory, 'target_image_%s.png' % steps))

                    self.save_progress(steps, np.mean(last_100_loss))

                    util.raw_score_plotter(train_loss)

            self.save_progress(e, np.mean(last_100_loss))

        image_to_gif('./img/', fpred_image_list, duration=1, gifname='fpred')
        image_to_gif('./img/', gpred_image_list, duration=1, gifname='gpred')
        image_to_gif('./img/', target_image_list, duration=1, gifname='target')
        image_to_gif('./img/', input_image_list, duration=1, gifname='input')


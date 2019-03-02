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

    def __init__(self, opts, trainloader, model, optimizer, criterion, checkpoint_path = './model/checkpoint_0.pth'):

        self.opts = opts
        self.trainloader = trainloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.checkpoint_path = checkpoint_path

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
        input_pred_list = []
        target_pred_list = []
        target_image_list = []
        input_image_list = []

        train_loss = []

        #save past images
        input_pred_deque = ReplayBuffer(50, 50, 0)
        target_pred_deque = ReplayBuffer(50, 50, 0)

        directory = './img/'

        epoch = 0

        if self.opts.resume:
            epoch, loss = self.load_progress()

        for e in range(self.opts.epoch - epoch):

            running_loss = 0

            for input_image, target_image in iter(self.trainloader):

                steps += 1

                input_image = input_image.to(device)
                target_image = target_image.to(device)

                '''Input(F) --> target(G)'''
                target_pred = self.model[0](input_image)
                target_pred_deque.add(target_pred.detach())

                '''G_discriminator'''
                '''Train Discriminator'''

                target_fake_logits = self.model[2](target_pred_deque.sample())
                target_real_logits = self.model[2](target_image)

                g_discrim_loss = util.discriminator_loss(target_real_logits, target_fake_logits)

                '''G_discriminator loss for the generator F'''

                target_gen_logit = self.model[2](target_pred)
                target_gen_loss = util.generator_loss(target_gen_logit)

                '''Input(G) --> target(F)'''
                input_pred = self.model[1](target_image)
                input_pred_deque.add(input_pred.detach())

                '''F_discriminator'''

                input_fake_logits = self.model[3](input_pred_deque.sample())
                input_real_logits = self.model[3](input_image)

                f_discrim_loss = util.discriminator_loss(input_real_logits, input_fake_logits)

                '''F_discriminator loss for the generator F'''
                input_gen_logits = self.model[3](input_pred)
                input_gen_loss = util.generator_loss(input_gen_logits)

                '''Cycle Consistency'''

                '''input -> target -> input Cycle'''
                I_T_I_pred = self.model[1](self.model[0](input_image))
                I_T_I_loss = self.criterion(I_T_I_pred, input_image)

                '''target -> input -> target Cycle'''
                T_I_T_pred = self.model[0](self.model[1](target_image))
                T_I_T_loss = self.criterion(T_I_T_pred, target_image)

                cycle_loss = I_T_I_loss + T_I_T_loss

                '''Discriminators'''
                '''G Discriminator'''
                '''Model Update'''
                self.optimizer[0].zero_grad()
                gen_loss = target_gen_loss + input_gen_loss + self.opts.cycle_lambda * cycle_loss
                gen_loss.backward()
                self.optimizer[0].step()

                '''input Discriminator'''
                self.optimizer[1].zero_grad()
                g_discrim_loss.backward()
                self.optimizer[1].step()

                '''target Discriminator'''
                self.optimizer[2].zero_grad()
                f_discrim_loss.backward()
                self.optimizer[2].step()

                last_100_loss.append(gen_loss.item())
                running_loss += gen_loss.item()
                train_loss.append(gen_loss.item())

                if steps % self.opts.print_every == 0:

                    print('\rEpoch {}\tSteps {}\tLoss: {:.4f}\n'.format(e, steps, np.mean(last_100_loss)), end="")

                    input_pred_list.append(util.save_images_to_directory(input_pred, directory, 'input_pred_image_%s.png' % steps))
                    target_pred_list.append(util.save_images_to_directory(target_pred, directory, 'target_pred_image_%s.png' % steps))
                    input_image_list.append(util.save_images_to_directory(input_image, directory, 'input_image_%s.png' % steps))
                    target_image_list.append(util.save_images_to_directory(target_image, directory, 'target_image_%s.png' % steps))

                    util.raw_score_plotter(train_loss)

            if self.opts.save_progress:
                self.save_progress(e, np.mean(last_100_loss))

        image_to_gif('./img/', input_pred_list, duration=1, gifname='input_pred')
        image_to_gif('./img/', target_pred_list, duration=1, gifname='target_pred')
        image_to_gif('./img/', target_image_list, duration=1, gifname='target')
        image_to_gif('./img/', input_image_list, duration=1, gifname='input')


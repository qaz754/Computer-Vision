
import torch
import os
import torch.nn as nn
import numpy as np

import util
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from image_to_gif import image_to_gif
import torch.nn.functional as F

from collections import deque


class GAN_Trainer():
    def __init__(self, opts, D, G, D_solver, G_solver, loader):

        self.opts = opts
        self.D = D
        self.G = G
        self.D_solver = D_solver
        self.G_solver = G_solver
        self.loader = loader
        self.checkpoint_path = './model/checkpoint_const1.pth'

    def save_progress(self, epoch, loss):

        directory = './model/'
        filename = 'checkpoint_%s.pth' % epoch

        path = os.path.join('%s' % directory, '%s' % filename)

        torch.save({
            'epoch': epoch,
            'loss': loss,
            'Discriminator_state_dict': self.D.state_dict(),
            'Generator_state_dict': self.G.state_dict(),
            'D_solver_state_dict': self.D_solver.state_dict(),
            'G_solver_state_dict' : self.G_solver.state_dict(),
        }, path)

        print("Saving Training Progress")

    def load_progress(self,):

        # TODO get rid of hardcoding and come up with an automated way.

        checkpoint = torch.load(self.checkpoint_path)

        self.D.load_state_dict(checkpoint['Discriminator_state_dict'])
        self.G.load_state_dict(checkpoint['Generator_state_dict'])

        self.D_solver.load_state_dict(checkpoint['D_solver_state_dict'])
        self.G_solver.load_state_dict(checkpoint['G_solver_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print("Training Loss From Last Session")
        return epoch, loss

    def train(self):
        """
        Vanilla GAN Trainer

        :param D: Discriminator
        :param G: Generator
        :param D_solver: Optimizer for D
        :param G_solver: Optimizer for G
        :param discriminator_loss:  Loss for D
        :param generator_loss:  Loss for G
        :param loader: Torch dataloader
        :param show_every: Show samples after every show_every iterations
        :param batch_size: Batch Size used for training
        :param noise_size: Dimension of the noise to use as input for G
        :param num_epochs: Number of epochs over the training dataset to use for training
        :return:
        """
        last_100_loss = deque(maxlen=100)
        last_100_g_loss =[]

        iter_count = 0
        filelist = []
        input_list = []

        last_epoch = 0
        if self.opts.resume:
            last_epoch, loss = self.load_progress()

        directory = './img/'

        for epoch in range(self.opts.epoch- last_epoch):

            '''Adaptive LR Change'''
            for param_group in self.D_solver.param_groups:
                param_group['lr'] = util.linear_LR(epoch, self.opts)
                print('epoch: {}, D_LR: {:.4}'.format(epoch, param_group['lr']))

            for param_group in self.G_solver.param_groups:
                param_group['lr'] = util.linear_LR(epoch, self.opts)
                print('epoch: {}, G_LR: {:.4}'.format(epoch, param_group['lr']))

            if self.opts.save_progress:
                '''Save the progress before start adjusting the LR'''
                if epoch == self.opts.const_epoch:
                    self.save_progress(self.opts.const_epoch, np.mean(last_100_loss))

            for image, label, target_label in self.loader:

                '''Real Images'''
                image = image.to(device)

                '''one hot encode the real label'''

                label = label.float().to(device)
                target_label = target_label.float().to(device)
                '''Train Discriminator'''
                '''Get the logits'''
                fake_images = self.G(image, target_label)

                fake_logits_src, _ = self.D(fake_images.detach())
                real_logits_src, real_logits_cls = self.D(image)

                D_cls_loss = self.opts.cls_lambda * F.binary_cross_entropy_with_logits(real_logits_cls, label, reduction='sum') / real_logits_cls.size(0)
                GP_loss = util.calc_gradient_penalty(self.D, image.data, fake_images.data) * self.opts.gp_lambda
                D_loss = -torch.mean(real_logits_src) + torch.mean(fake_logits_src) + D_cls_loss + GP_loss

                self.D_solver.zero_grad()
                D_loss.backward()
                self.D_solver.step()  # One step Descent into loss

                '''Train Generator'''
                iter_count += 1
                if iter_count % self.opts.n_critic == 0:

                    fake_image = self.G(image, target_label)
                    fake_logits_src, fake_logits_cls = self.D(fake_image)
                    reconstruction = self.G(fake_image, label)

                    '''Reconstruction'''
                    G_cls_loss = self.opts.cls_lambda * F.binary_cross_entropy_with_logits(fake_logits_cls, target_label, reduction='sum') / fake_logits_cls.size(0)
                    recon_loss = nn.L1Loss()(reconstruction, target=image) * self.opts.cycle_lambda

                    G_loss = -torch.mean(fake_logits_src) + G_cls_loss + recon_loss

                    #plot error
                    last_100_loss.append(G_loss.item())
                    last_100_g_loss.append(np.mean(last_100_loss))
                    util.raw_score_plotter(last_100_g_loss)

                    self.G_solver.zero_grad()
                    G_loss.backward()
                    self.G_solver.step()  # One step Descent into the loss

                if iter_count % self.opts.print_every == 0:
                    print('Epoch: {}, Iter: {}, D: {:.4}, D_cls: {:.4}, D_GP: {:.4} G: {:.4} G_cls:{:.4}'.format(epoch, iter_count, D_loss.item(), D_cls_loss.item(), GP_loss.item(), G_loss.item(), G_cls_loss.item()))

                if iter_count % self.opts.show_every == 0:
                    gen_image = util.stargan_side_by_side_images(self.opts, self.G, image, label)
                    filelist.append(util.save_images_to_directory(gen_image, directory, 'generated_image_%s.png' % iter_count, nrow=1))

                if self.opts.save_progress:
                    if iter_count % self.opts.save_every == 0:
                        self.save_progress(epoch, np.mean(last_100_loss))

        if self.opts.save_progress:
            '''Save the progress before start adjusting the LR'''
            self.save_progress(-1, np.mean(last_100_loss))

        #create a gif
        image_to_gif('./img/', filelist, duration=1, gifname='transformed')





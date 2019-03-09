
import torch
import os
import torch.nn as nn
import numpy as np

import util
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch.nn.functional as F

from image_to_gif import image_to_gif

TARGET_LABEL = 8

from collections import deque

def run_vanilla_gan(opts, D, G, D_solver, G_solver, loader):
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

    directory = './img/'

    for epoch in range(opts.epoch):

        for param_group in D_solver.param_groups:
            param_group['lr'] = util.linear_LR(epoch, opts)
            print('Epoch: {}, D_LR: {:.4}'.format(epoch, param_group['lr']))

        for param_group in G_solver.param_groups:
            param_group['lr'] = util.linear_LR(epoch, opts)
            print('Epoch: {}, G_LR: {:.4}'.format(epoch, param_group['lr']))

        for image, label in loader:

            '''Real Images'''
            image = image.to(device)

            '''one hot encode the real label'''

            label_onehot = util.one_hot_encoder(label, n_classes=opts.num_classes)
            label_onehot = torch.from_numpy(label_onehot).float().to(device)

            target_label = np.ones(label.shape[0], dtype=int) * TARGET_LABEL
            target_domain = util.one_hot_encoder(target_label, n_classes=opts.num_classes)
            target_domain = torch.from_numpy(target_domain).float().to(device)

            '''Train Discriminator'''
            '''Get the logits'''

            fake_images = G(image, target_domain)

            fake_logits_src, _ = D(fake_images.detach())
            real_logits_src, real_logits_cls = D(image)
            D_cls_loss = opts.cls_lambda * F.binary_cross_entropy_with_logits(real_logits_cls, label_onehot, reduction='sum') / real_logits_cls.size(0)
            GP_loss = util.calc_gradient_penalty(D, image.data, fake_images.data) * opts.gp_lambda
            D_loss = -torch.mean(real_logits_src) + torch.mean(fake_logits_src) + D_cls_loss + GP_loss

            D_solver.zero_grad()
            D_loss.backward()
            D_solver.step()  # One step Descent into loss

            '''Train Generator'''

            iter_count += 1

            if iter_count % opts.n_critic == 0:

                fake_image = G(image, target_domain)
                fake_logits_src, fake_logits_cls = D(fake_image)
                reconstruction = G(fake_image, label_onehot)

                '''Reconstruction'''
                G_cls_loss = opts.cls_lambda * F.binary_cross_entropy_with_logits(fake_logits_cls, target_domain, reduction='sum') / fake_logits_cls.size(0)
                recon_loss = nn.L1Loss()(reconstruction, target=image) * opts.cycle_lambda

                G_loss = -torch.mean(fake_logits_src) + G_cls_loss + recon_loss

                #plot error
                last_100_loss.append(G_loss.item())
                last_100_g_loss.append(np.mean(last_100_loss))
                util.raw_score_plotter(last_100_g_loss)

                G_solver.zero_grad()
                G_loss.backward()
                G_solver.step()  # One step Descent into the loss

            if iter_count % opts.print_every == 0:
                print('Epoch: {}, Iter: {}, D: {:.4}, D_cls: {:.4}, D_GP: {:.4} G: {:.4} G_cls:{:.4}'.format(epoch, iter_count, D_loss.item(), D_cls_loss.item(), GP_loss.item(), G_loss.item(), G_cls_loss.item()))

            if iter_count % opts.show_every == 0:
                input_list.append(util.save_images_to_directory(image.view((image.shape)), directory, 'input_image_%s.png' % iter_count))
                filelist.append(util.save_images_to_directory(fake_images.view((image.shape)), directory, 'generated_image_%s.png' % iter_count))

    #create a gif
    image_to_gif('./img/', filelist, duration=1, gifname='transformed')
    image_to_gif('./img/', input_list, duration=1, gifname='original')


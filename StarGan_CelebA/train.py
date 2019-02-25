
import torch
import os
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

from util import sample_noise, show_images, one_hot_encoder, categorical_label_generator, Flatten, save_images_to_directory
from network import Flatten
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt

from image_to_gif import image_to_gif

def run_vanilla_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, loader, show_every=100, batch_size=128, noise_size=96, n_classes=10, num_epochs = 10):
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

    iter_count = 0
    target_list = []
    input_list = []
    lamb_rec = 10
    lamb_cls = 1

    for epoch in range(num_epochs):
        for image, label, target_label in loader:

            '''Real Images'''
            image = image.to(device)

            '''one hot encode the real label'''
            label_e = label.view((label.shape[0], label.shape[1], 1, 1))
            label_e = label_e.repeat(1, 1, 128, 128).float().to(device)
            label = label.float().to(device)

            target_domain_e = target_label.view((target_label.shape[0], target_label.shape[1], 1, 1))
            target_domain_e = target_domain_e.repeat(1, 1, 128, 128).float().to(device)

            target_label = target_label.float().to(device)

            fake_images = G(torch.cat((image, target_domain_e), dim=1))
            fake_logits_src, fake_logits_cls = D(fake_images)

            '''Train Generator'''
            G_cls_loss = F.binary_cross_entropy_with_logits(fake_logits_cls, target_label, size_average=False) / fake_logits_cls.size(0)
            G_src_loss = generator_loss(fake_logits_src)

            '''Reconstruction'''
            reconstruction = G(torch.cat((fake_images, label_e), dim=1))
            recon_loss = nn.L1Loss()(reconstruction, target=image)

            G_loss = G_src_loss + lamb_cls * G_cls_loss + lamb_rec * recon_loss

            G_solver.zero_grad()
            G_loss.backward()
            G_solver.step()  # One step Descent into the loss

            '''Train Discriminator'''

            fake_images_D = G(torch.cat((image, target_domain_e), dim=1))

            fake_logits_src, fake_logits_cls = D(fake_images_D)
            real_logits_src, real_logits_cls = D(image)

            D_src_loss = discriminator_loss(real_logits_src, fake_logits_src)
            D_cls_loss = F.binary_cross_entropy_with_logits(real_logits_cls, label, size_average=False) / real_logits_cls.size(0)

            D_loss = D_src_loss + D_cls_loss

            D_solver.zero_grad()
            D_loss.backward()
            D_solver.step() #One step Descent into loss

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, D_loss.item(), G_loss.item()))

                directory = './img/'

                input_list.append(save_images_to_directory(image, directory, 'input_image_%s.png' % iter_count))
                target_list.append(save_images_to_directory(fake_images, directory, 'target_image_%s.png' % iter_count))

            iter_count += 1

    #create a gif
    image_to_gif('./img/', input_list, duration=1)
    image_to_gif('./img/', target_list, duration=1)


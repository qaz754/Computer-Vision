
import torch
import os
import torch.nn as nn

import util
from network import Flatten
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt

import torch.nn.functional as F

from image_to_gif import image_to_gif

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

    iter_count = 0
    filelist = []
    directory = './img/'
    Flattener = Flatten()


    for epoch in range(opts.epoch):
        for image, label in loader:

            '''Real Images'''
            image = image.to(device)

            label = label.to(device)

            '''one hot encode the real label'''
            labels = util.one_hot_encoder(label, n_classes=opts.num_classes)
            labels = torch.from_numpy(labels).float().to(device)

            g_fake_seed = util.sample_noise(label.shape[0], opts.noise_dim).to(device)
            fake_images = G(torch.cat((g_fake_seed, labels), dim=1))

            fake_logits_src, fake_logits_cls = D(fake_images)

            '''Train Generator'''
            G_cls_loss = nn.CrossEntropyLoss()(fake_logits_cls, label)
            G_src_loss = util.generator_loss(fake_logits_src)

            G_loss = G_src_loss + G_cls_loss

            G_solver.zero_grad()
            G_loss.backward()
            G_solver.step()  # One step Descent into the loss

            '''Train Discriminator'''
            g_fake_seed = util.sample_noise(label.shape[0], opts.noise_dim).to(device)
            fake_images = G(torch.cat((g_fake_seed, labels), dim=1))

            fake_logits_src, fake_logits_cls = D(fake_images)
            real_logits_src, real_logits_cls = D(Flattener(image))

            D_src_loss = util.discriminator_loss(real_logits_src, fake_logits_src)
            D_cls_loss = nn.CrossEntropyLoss()(real_logits_cls, label)

            D_loss = D_src_loss + D_cls_loss

            D_solver.zero_grad()
            D_loss.backward()
            D_solver.step() #One step Descent into loss

            if iter_count % opts.print_every == 0:
                print('Epoch: {}, Iter: {}, D: {:.4}, G:{:.4}'.format(epoch, iter_count, D_loss.item(), G_loss.item()))

            if iter_count % opts.show_every == 0:
                filelist.append(util.save_images_to_directory(fake_images.view((image.shape)), directory, 'generated_image_%s.png' % iter_count))

            iter_count += 1

    #create a gif
    image_to_gif('./img/', filelist, duration=1)


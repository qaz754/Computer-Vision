
import torch
import os
import torch.nn as nn
import numpy as np

import util
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt

from image_to_gif import image_to_gif

TARGET_LABEL = 8

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
    input_list = []

    directory = './img/'

    for epoch in range(opts.epoch):
        for image, label in loader:

            '''Real Images'''
            image = image.to(device)

            '''one hot encode the real label'''

            label_onehot = util.one_hot_encoder(label, n_classes=opts.num_classes)
            label_onehot = torch.from_numpy(label_onehot).float().to(device)

            label_e = util.expand_spatially(label_onehot, 28).float().to(device)

            target_label = np.ones(label.shape[0], dtype=int) * TARGET_LABEL
            target_domain = util.one_hot_encoder(target_label, n_classes=opts.num_classes)
            target_domain = torch.from_numpy(target_domain).float().to(device)

            target_domain_e = util.expand_spatially(target_domain, 28).float().to(device)
            fake_images = G(torch.cat((image, target_domain_e), dim=1))
            fake_logits_src, fake_logits_cls = D(fake_images)

            '''Train Generator'''
            G_cls_loss = nn.BCEWithLogitsLoss()(fake_logits_cls, target=target_domain)
            G_src_loss = util.generator_loss(fake_logits_src)

            '''Reconstruction'''
            reconstruction = G(torch.cat((fake_images, label_e), dim=1))
            recon_loss = nn.L1Loss()(reconstruction, target=image)

            G_loss = G_src_loss + opts.cls_lambda * G_cls_loss + opts.cycle_lambda * recon_loss

            G_solver.zero_grad()
            G_loss.backward()
            G_solver.step()  # One step Descent into the loss

            '''Train Discriminator'''

            fake_images_D = G(torch.cat((image, target_domain_e), dim=1))

            fake_logits_src, fake_logits_cls = D(fake_images_D)
            real_logits_src, real_logits_cls = D(image)

            D_src_loss = util.discriminator_loss(real_logits_src, fake_logits_src)
            D_cls_loss = nn.BCEWithLogitsLoss()(real_logits_cls, target=label_onehot)

            D_loss = D_src_loss + opts.cls_lambda * D_cls_loss

            D_solver.zero_grad()
            D_loss.backward()
            D_solver.step() #One step Descent into loss

            if iter_count % opts.print_every == 0:
                print('Epoch: {}, Iter: {}, D: {:.4}, G:{:.4}'.format(epoch, iter_count, D_loss.item(), G_loss.item()))

            if iter_count % opts.show_every == 0:
                input_list.append(util.save_images_to_directory(image.view((image.shape)), directory, 'input_image_%s.png' % iter_count))
                filelist.append(util.save_images_to_directory(fake_images.view((image.shape)), directory, 'generated_image_%s.png' % iter_count))

            iter_count += 1

    #create a gif
    image_to_gif('./img/', filelist, duration=1, gifname='transformed')
    image_to_gif('./img/', input_list, duration=1, gifname='original')


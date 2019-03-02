
import torch
import os

import util

from util import sample_noise, show_images
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt

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


    for epoch in range(opts.epoch):
        for x, _ in loader:

            '''Real Images'''
            real_data = x.view((-1, opts.D_input_size)).to(device)

            '''Train Generator'''
            g_fake_seed = sample_noise(len(real_data), opts.noise_dim).to(device)
            fake_images = G(g_fake_seed).view(-1, opts.D_input_size)

            gen_logits_fake = D(fake_images)
            g_error = util.generator_loss(gen_logits_fake)

            G_solver.zero_grad()
            g_error.backward()
            G_solver.step()

            '''Train Discriminator'''
            logits_real = D(real_data).to(device)
            logits_fake = D(fake_images.detach())
            d_total_error = util.discriminator_loss(logits_real, logits_fake)

            D_solver.zero_grad()
            d_total_error.backward()
            D_solver.step()

            if (iter_count % opts.print_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.item(), g_error.item()))

            if (iter_count % opts.show_every == 0):
                imgs_numpy = fake_images.view(x.shape).data.cpu().numpy()

                '''filename used for saving the image'''
                filelist.append(util.save_images_to_directory(imgs_numpy, directory, 'generated_image_%s.png' % iter_count))

            iter_count += 1

    #create a gif
    image_to_gif('./img/', filelist, duration=1)

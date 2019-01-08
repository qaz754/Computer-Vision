
import torch
import os

from util import sample_noise, show_images, get_patches, patch_discrim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from torchvision.utils import save_image

from image_to_gif import image_to_gif

def run_vanilla_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, loader, show_every=250, batch_size=128, noise_size=96, num_epochs = 10):
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


    for epoch in range(num_epochs):
        for x, _ in loader:
            if len(x) != batch_size:
                continue
            D_solver.zero_grad()

            kernel_window = 26

            '''Real Images'''
            real_data = x.to(device) #sampled batch of data
            real_data = 2 * (real_data - 0.5)
            real_patches = get_patches(real_data, kernel_window)

            logits_real = patch_discrim(real_patches, batch_size, D)
            '''Train Discriminator'''

            g_fake_seed = sample_noise(batch_size, noise_size).to(device) #Sample minibatch of m noise samples
            fake_images = G(g_fake_seed).view(batch_size, 1, 28, 28).detach() #Sample minibatch of m examples from data generating distribution

            fake_patches = get_patches(fake_images, kernel_window)
            logits_fake = patch_discrim(fake_patches, batch_size, D)

            d_total_error = discriminator_loss(logits_real, logits_fake) #negative Sigmoid BCE loss for the discriminator
            d_total_error.backward()
            D_solver.step() #One step Descent into loss

            '''Train Generator'''
            G_solver.zero_grad()
            g_fake_seed = sample_noise(batch_size, noise_size).to(device) #Sample minibatch of m noise samples from noise prior
            fake_images = G(g_fake_seed).view(batch_size, 1, 28, 28) #Sample minibatch of m examples from data generating distribution

            fake_patches = get_patches(fake_images, kernel_window)
            gen_logits_fake = patch_discrim(fake_patches, batch_size, D)

            g_error = generator_loss(gen_logits_fake) #get the loss for the generator
            g_error.backward()
            G_solver.step() #One step Descent into the loss

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.item(), g_error.item()))
                imgs_numpy = fake_images.data.cpu().numpy()

                '''filename used for saving the image'''
                directory = './img/'
                filename = 'image_%s.png' %iter_count
                filelist.append(filename)

                filename = os.path.join('%s' % directory, '%s' % filename)


                show_images(imgs_numpy[0:16], filename, iter_count)
                plt.show()
                print()



            iter_count += 1

    #create a gif
    image_to_gif('./img/', filelist, duration=1)


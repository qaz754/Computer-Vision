
import torch


import util
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from image_to_gif import image_to_gif

def run_vanilla_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, loader, show_every=250, batch_size=32, noise_size=96, num_epochs = 10):
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
    x_filelist = []
    y_filelist = []
    x_real_filelist = []
    y_real_filelist = []
    directory = './img/'

    for epoch in range(num_epochs):
        for x, y in loader:

            '''Real Images'''
            x = x.to(device)
            y = y.to(device)

            g_fake_seed = util.sample_noise(y.shape[0], noise_size).to(device)

            fake_image_x, fake_image_y = G(g_fake_seed)
            logits_fake_image_x, logits_fake_image_y = D(fake_image_x, fake_image_y)
            g_error = util.generator_loss(logits_fake_image_x) + util.generator_loss(logits_fake_image_y)
            g_error.to(device)
            G_solver.zero_grad()
            g_error.backward()
            G_solver.step()  # One step Descent into the loss

            '''Train Discriminator'''
            g_fake_seed = util.sample_noise(y.shape[0], noise_size).to(device) #Sample minibatch of m noise samples
            fake_image_x, fake_image_y = G(g_fake_seed) #Sample minibatch of m examples from data generating distribution

            logits_fake_image_x, logits_fake_image_y = D(fake_image_x.detach(), fake_image_y.detach())
            logits_real_image_x, logits_real_image_y = D(x, y)

            d_total_error = util.discriminator_loss(logits_real_image_x, logits_fake_image_x) + util.discriminator_loss(logits_real_image_y, logits_fake_image_y)
            d_total_error = d_total_error.to(device)
            D_solver.zero_grad()
            d_total_error.backward()
            D_solver.step() #One step Descent into loss

            if (iter_count % show_every == 0):

                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.item(), g_error.item()))
                '''filename used for saving the image'''

                x_filelist.append(
                    util.save_images_to_directory(fake_image_x, directory, 'x_gen_image_%s.png' % iter_count))

                y_filelist.append(
                    util.save_images_to_directory(fake_image_y, directory, 'y_gen_image_%s.png' % iter_count))

                x_real_filelist.append(
                    util.save_images_to_directory(x, directory, 'x_image_%s.png' % iter_count))

                y_real_filelist.append(
                    util.save_images_to_directory(y, directory, 'y_image_%s.png' % iter_count))

            iter_count += 1

    #create a gif
    image_to_gif('./img/', x_filelist, duration=.5, gifname='x_pred')
    image_to_gif('./img/', y_filelist, duration=.5, gifname='y_pred')

    image_to_gif('./img/', x_real_filelist, duration=.5, gifname='x_real')
    image_to_gif('./img/', y_real_filelist, duration=.5, gifname='y_real')



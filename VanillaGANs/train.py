
import torch
from util import sample_noise, show_images, discriminator_loss, generator_loss
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def run_vanilla_gan(D, G, D_solver, G_solver, loader, show_every=250, batch_size=128, noise_size=96, num_epochs = 10):
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
    images = []

    for epoch in range(num_epochs):
        for x, _ in loader:
            if len(x) != batch_size:
                continue
            D_solver.zero_grad()

            '''Real Images'''
            real_data = x.to(device)
            logits_real = D(2 * (real_data - 0.5)).to(device)

            '''Train Discriminator'''
            g_fake_seed = sample_noise(batch_size, noise_size).to(device)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images.view(batch_size, 1, 28, 28))

            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()
            D_solver.step()

            '''Train Generator'''
            G_solver.zero_grad()
            g_fake_seed = sample_noise(batch_size, noise_size).to(device)
            fake_images = G(g_fake_seed)

            gen_logits_fake = D(fake_images.view(batch_size, 1, 28, 28))
            g_error = generator_loss(gen_logits_fake)
            g_error.backward()
            G_solver.step()

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.item(), g_error.item()))
                imgs_numpy = fake_images.data.cpu().numpy()
                show_images(imgs_numpy[0:16])
                plt.show()
                print()
            iter_count += 1



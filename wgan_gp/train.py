
import torch


import util
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            real_data = x.view((-1, 784)).to(device)
            logits_real = D(real_data).to(device)

            '''Train Discriminator'''
            g_fake_seed = util.sample_noise(x.shape[0], opts.noise_dim).to(device)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images)

            d_total_error = -torch.mean(logits_real) + torch.mean(logits_fake) + util.calc_gradient_penalty(D, real_data, fake_images) * opts.gp_lambda

            D_solver.zero_grad()
            d_total_error.backward()
            D_solver.step()

            '''Train Generator Every n_critics iterations'''
            if iter_count % opts.n_critic == 0:

                g_fake_seed = util.sample_noise(x.shape[0], opts.noise_dim).to(device)
                fake_images = G(g_fake_seed)

                gen_logits_fake = D(fake_images)
                g_error = -torch.mean(gen_logits_fake)

                G_solver.zero_grad()
                g_error.backward()
                G_solver.step()

            if iter_count % opts.print_every == 0:
                print('Epoch: {}, Iter: {}, D: {:.4}, G:{:.4}'.format(epoch, iter_count, d_total_error.item(),
                                                                      g_error.item()))

            if iter_count % opts.show_every == 0:

                filelist.append(
                    util.save_images_to_directory(fake_images.view((x.shape)), directory, 'target_image_%s.png' % iter_count))

            iter_count += 1

    #create a gif
    image_to_gif('./img/', filelist, duration=0.5)


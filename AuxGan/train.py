
import torch
import os
import torch.nn as nn

from util import sample_noise, show_images, one_hot_encoder, categorical_label_generator, Flatten
from network import Flatten
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from torchvision.utils import save_image

from image_to_gif import image_to_gif

def run_vanilla_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, loader, show_every=250, batch_size=128, noise_size=96, n_classes=10, num_epochs = 10):
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

    Flattener = Flatten()


    for epoch in range(num_epochs):
        for image, label in loader:

            '''Real Images'''
            image = 2 * (image - 0.5)
            image = image.to(device)

            '''one hot encode the real label'''
            label = one_hot_encoder(label, n_classes=n_classes)
            label = torch.from_numpy(label).float().to(device)

            g_fake_seed = sample_noise(label.shape[0], noise_size).to(device)
            fake_images = G(torch.cat((g_fake_seed, label), dim=1))

            fake_logits_src, fake_logits_cls = D(fake_images)

            '''Train Generator'''

            G_cls_loss = nn.MSELoss()(fake_logits_cls, target=label)
            G_src_loss = generator_loss(fake_logits_src)


            G_loss = G_src_loss + G_cls_loss

            G_solver.zero_grad()
            G_loss.backward()
            G_solver.step()  # One step Descent into the loss

            '''Train Discriminator'''

            g_fake_seed = sample_noise(label.shape[0], noise_size).to(device)
            fake_images = G(torch.cat((g_fake_seed, label), dim=1))

            fake_logits_src, fake_logits_cls = D(fake_images)
            real_logits_src, real_logits_cls = D(Flattener(image))

            D_src_loss = discriminator_loss(real_logits_src, fake_logits_src)
            D_cls_loss = nn.MSELoss()(real_logits_cls, target=label)

            D_loss = D_src_loss + D_cls_loss

            D_solver.zero_grad()
            D_loss.backward()
            D_solver.step() #One step Descent into loss


            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, D_loss.item(), G_loss.item()))
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



import torch
import os
import torch.nn as nn
import numpy as np

from util import sample_noise, show_images, one_hot_encoder, categorical_label_generator, Flatten
from network import Flatten
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt

from image_to_gif import image_to_gif

TARGET_LABEL = 8

def run_vanilla_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, loader, show_every=750, batch_size=128, noise_size=96, n_classes=10, num_epochs = 10):
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
    lamb_rec = 10
    lamb_cls = 1

    for epoch in range(num_epochs):
        for image, label in loader:

            '''Real Images'''
            image = 2 * (image - 0.5)
            image = image.to(device)

            '''one hot encode the real label'''

            label_onehot = one_hot_encoder(label, n_classes=n_classes)
            label_onehot = torch.from_numpy(label_onehot).float().to(device)
            label_e = label_onehot.view((label_onehot.shape[0], label_onehot.shape[1], 1, 1))
            label_e = label_e.repeat(1, 1, 28, 28).float().to(device)
            label = label.to(device)

            target_label = np.ones(label.shape[0], dtype=int) * TARGET_LABEL
            target_domain = one_hot_encoder(target_label, n_classes=n_classes)
            target_domain = torch.from_numpy(target_domain).float().to(device)

            target_domain_e = target_domain.view((target_domain.shape[0], target_domain.shape[1], 1, 1))
            target_domain_e = target_domain_e.repeat(1, 1, 28, 28).float().to(device)

            target_label = torch.from_numpy(target_label).long().to(device)

            fake_images = G(torch.cat((image, target_domain_e), dim=1))
            fake_logits_src, fake_logits_cls = D(fake_images)

            '''Train Generator'''
            G_cls_loss = nn.CrossEntropyLoss()(fake_logits_cls, target=target_label)
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
            D_cls_loss = nn.CrossEntropyLoss()(real_logits_cls, target=label)


            D_loss = D_src_loss + D_cls_loss

            D_solver.zero_grad()
            D_loss.backward()
            D_solver.step() #One step Descent into loss

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, D_loss.item(), G_loss.item()))

                directory = './img/'

                filename = 'Input_Image_%s.png' % iter_count
                input_list.append(filename)

                filename = os.path.join('%s' % directory, '%s' % filename)
                show_images(image[0:16], filename, iter_count)
                plt.show()
                print()

                imgs_numpy = fake_images.data.cpu().numpy()

                '''filename used for saving the image'''
                filename = 'image_%s.png' %iter_count
                filelist.append(filename)

                filename = os.path.join('%s' % directory, '%s' % filename)

                show_images(imgs_numpy[0:16], filename, iter_count)
                plt.show()
                print()

            iter_count += 1

    #create a gif
    image_to_gif('./img/', filelist, duration=1)
    image_to_gif('./img/', input_list, duration=1)



import torch
import os

from util import sample_noise, show_images, one_hot_encoder, categorical_label_generator
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from torchvision.utils import save_image

from image_to_gif import image_to_gif

def run_vanilla_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, loader, show_every=250, batch_size=128, noise_size=96, num_classes = 10, num_epochs = 10):
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
        for x, real_label in loader:
            if len(x) != batch_size:
                continue
            D_solver.zero_grad()

            '''Real Images'''
            real_data = x.to(device) #sampled batch of data
            #real_label = one_hot_encoder(real_label, num_classes)
            #real_label = torch.from_numpy(real_label).long().to(device)
            real_label = real_label.long().to(device)

            logits_real = D(2 * (real_data - 0.5)).to(device) #returns logit of real data.

            #values, indices = logits_real.max(1)

            #indices = one_hot_encoder(indices, num_classes)
            #indices = torch.from_numpy(indices).float().to(device)
            #indices = (indices).float().to(device)

            #TODO indicies hold the index that has max logit
            '''Train Discriminator'''

            g_fake_seed = sample_noise(batch_size, noise_size).to(device) #Sample minibatch of m noise samples

            fake_label = categorical_label_generator(batch_size=batch_size, n_classes=num_classes)
            #fake_label_encoded = one_hot_encoder(fake_label, num_classes)
            fake_label_encoded = torch.from_numpy(fake_label).long().to(device)

            fake_label = torch.from_numpy(fake_label).long().to(device)

            g_fake_seed = torch.cat((g_fake_seed, fake_label.float().unsqueeze(1)), dim=1)

            fake_images = G(g_fake_seed).detach() #Sample minibatch of m examples from data generating distribution
            logits_fake = D(fake_images.view(batch_size, 1, 28, 28)) #get the logits for the fake images using the Discirminator

            #fake_values, fake_indices = logits_fake.max(1)

            #fake_indices = one_hot_encoder(fake_indices, num_classes)
            #fake_indices = torch.from_numpy(fake_indices).float().to(device)
            #fake_indices = (fake_indices).float().to(device)

            labels = torch.cat((real_label, fake_label))

            '''Get the predicted labels for both real and fake data'''
            predicted_labels = torch.cat((logits_real, logits_fake))

            d_total_error = discriminator_loss(predicted_labels, labels) #negative Sigmoid BCE loss for the discriminator
            d_total_error.backward()
            D_solver.step() #One step Descent into loss

            '''Train Generator'''
            G_solver.zero_grad()
            fake_label = categorical_label_generator(batch_size=batch_size, n_classes=num_classes)
            g_fake_seed = sample_noise(batch_size, noise_size).to(device) #Sample minibatch of m noise samples

            fake_label = torch.from_numpy(fake_label).long().to(device)
            g_fake_seed = torch.cat((g_fake_seed, fake_label.float().unsqueeze(1)), dim=1)

            fake_images = G(g_fake_seed) #Sample minibatch of m examples from data generating distribution

            gen_logits_fake = D(fake_images.view(batch_size, 1, 28, 28)) #get the negative sigmoid BCE loss for the discriminator
            g_error = - discriminator_loss(gen_logits_fake, fake_label) #get the loss for the generator
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

                torch.save(G.state_dict(), os.path.join('./models', 'Generator-%d.pkl' % epoch))
                torch.save(D.state_dict(), os.path.join('./models', 'Discriminator-%d.pkl' % epoch))



            iter_count += 1

    #create a gif
    image_to_gif('./img/', filelist, duration=1)


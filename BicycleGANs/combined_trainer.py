
import util
import os

from image_to_gif import image_to_gif

import torch
from torchvision.utils import save_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class trainer():

    def __init__(self, epochs, trainloader, model, optimizer, criterion, print_every=5, lambd_l1=1, lamb_kl=1, lamb_latent=1):

        self.epochs = epochs
        self.trainloader = trainloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.print_every = print_every
        self.lambd = lambd_l1
        self.kl_lamb = lamb_kl
        self.lamb_latent = lamb_latent

    def train(self):
        steps = 0

        #used to make gifs later
        pred_image_list = []
        target_image_list = []
        input_image_list = []

        train_loss = []

        for e in range(self.epochs):
            running_loss = 0

            for input_image, target_image in iter(self.trainloader):
                steps += 1

                input_image = input_image.to(device)
                target_image = target_image.to(device)

                '''
                VAE
                '''

                '''Encoder Phase'''
                mu, sigma, Q_zb = self.model[3].forward(target_image)

                KL_loss = torch.sum(1 + sigma - mu ** 2 - sigma.exp()) * -0.5 * self.kl_lamb
                '''Encoder Phase'''

                image_pred = self.model[0](input_image, Q_zb)

                fake_logits = self.model[1](image_pred.detach())
                real_logits = self.model[1](target_image)

                cVAE_discrim_loss = util.discriminator_loss(real_logits, fake_logits)

                '''Regularization Phase (Discriminator)'''

                '''--------Reconstruction Phase--------'''
                cVAE_gen_logits = self.model[1](image_pred)
                cVAE_gen_loss = util.generator_loss(cVAE_gen_logits)

                image_pred = self.model[0](input_image, Q_zb)
                recon_loss = self.criterion(image_pred, target_image)




                '''Regularization Phase (Discriminator)'''

                #cLR-GAN discrim

                input_noise = torch.randn(input_image.shape[0], 8).to(device)

                image_pred = self.model[0](input_image, input_noise)

                fake_logits = self.model[2](image_pred.detach())
                real_logits = self.model[2](target_image)

                cLR_discrim_loss = util.discriminator_loss(real_logits, fake_logits)

                '''Regularization Phase (Discriminator)'''
                # cLR-GAN gen
                '''--------Reconstruction Phase--------'''

                gen_logits = self.model[2](image_pred)
                gen_loss = util.generator_loss(gen_logits)

                cLR_Gen_Loss = gen_loss

                '''--------Reconstruction Phase--------'''


                '''Update'''

                '''Generator Loss'''
                self.optimizer[3].zero_grad()
                self.optimizer[0].zero_grad()
                Generator_loss = cVAE_gen_loss + self.lambd * recon_loss + KL_loss + cLR_Gen_Loss
                Generator_loss.backward(retain_graph=True)
                self.optimizer[3].step()
                self.optimizer[0].step()
                '''Generator Loss'''

                '''cLR Encoder'''
                self.optimizer[0].zero_grad()
                _, _, Q_zb = self.model[3].forward(image_pred)
                latent_code_loss = self.lamb_latent * self.criterion(Q_zb, input_noise)
                latent_code_loss.backward()
                self.optimizer[0].step()
                '''cLR Encoder'''

                '''Discriminators cVAE'''
                self.optimizer[1].zero_grad()
                cVAE_discrim_loss.backward()
                self.optimizer[1].step()
                '''Discriminators cVAE'''

                '''Discriminators cLR'''
                self.optimizer[2].zero_grad()
                cLR_discrim_loss.backward()
                self.optimizer[2].step()
                '''Discriminators cLR'''

                running_loss += Generator_loss.item()
                train_loss.append(Generator_loss.item())

                if steps % self.print_every == 0:
                    print("Epoch: {}/{}...".format(e + 1, self.epochs),
                          "LossL {:.4f}".format(running_loss))

                running_loss = 0

            if e % 1 == 0:

                pred = image_pred.cpu().data
                directory = './img/'
                filename = 'pred_image_%s.png' % e
                pred_image_list.append(filename)

                filename = os.path.join('%s' % directory, '%s' % filename)
                save_image(pred, filename)

                target = target_image.cpu().data
                directory = './img/'
                filename = 'real_image%s.png' % e
                target_image_list.append(filename)

                filename = os.path.join('%s' % directory, '%s' % filename)
                save_image(target, filename)

                input = input_image.cpu().data
                directory = './img/'
                filename = 'Input_Image_%s.png' % e
                input_image_list.append(filename)

                filename = os.path.join('%s' % directory, '%s' % filename)
                save_image(input, filename)

                torch.save(self.model[0].state_dict(), './pix2pix_G.pth')
                torch.save(self.model[1].state_dict(), './pix2pix_cVAE_D.pth')
                torch.save(self.model[2].state_dict(), './pix2pix_cLR_D.pth')
                torch.save(self.model[3].state_dict(), './pix2pix_E.pth')

                util.raw_score_plotter(train_loss)

        image_to_gif('./img/', pred_image_list, duration=1, gifname='pred')
        image_to_gif('./img/', target_image_list, duration=1, gifname='target')
        image_to_gif('./img/', input_image_list, duration=1, gifname='input')


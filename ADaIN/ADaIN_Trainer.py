import util
import os
import numpy as np
from image_to_gif import image_to_gif
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from collections import deque

class trainer():

    def __init__(self, epochs, trainloader, model, optimizer, criterion, print_every=1500, lambda0=10, checkpoint_path = './model/checkpoint_0.pth', resume=False):

        self.epochs = epochs
        self.trainloader = trainloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.print_every = print_every

        self.lambda0 = lambda0

        self.checkpoint_path = checkpoint_path
        self.resume = resume

    def save_progress(self, epoch, loss):

        #TODO get rid of hardcoding and come up with an automated way.
        directory = './model/'
        filename = 'checkpoint_%s.pth' % epoch

        path = os.path.join('%s' % directory, '%s' % filename)

        torch.save({
            'epoch': epoch,
            'loss': loss,
            'model2_state_dict': self.model[2].state_dict(),
            'optimizer1_state_dict': self.optimizer[0].state_dict(),
        }, path)

        print("Saving Training Progress")

    def load_progress(self,):

        # TODO get rid of hardcoding and come up with an automated way.

        checkpoint = torch.load(self.checkpoint_path)

        self.model[2].load_state_dict(checkpoint['model2_state_dict'])
        self.optimizer[0].load_state_dict(checkpoint['optimizer1_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print("Training Loss From Last Session")
        return epoch, loss

    def train(self):
        steps = 0

        last_100_loss = deque(maxlen=100)
        # used to make gifs later
        fpred_image_list = []
        target_image_list = []
        input_image_list = []

        train_loss = []

        directory = './img/'

        epoch = 0

        if self.resume:
            epoch, loss = self.load_progress()

        for e in range(self.epochs - epoch):

            running_loss = 0

            for content, style in iter(self.trainloader):

                steps += 1

                content = content.to(device)
                style = style.to(device)

                content_enc = self.model[0](content)
                style_enc = self.model[0](style)

                ADaIn_out = self.model[1].ADaIN([content_enc[-1], style_enc[-1]])
                transformed_image = self.model[2](ADaIn_out, content_enc[-1])

                '''
                Content Loss
                '''
                transformed_enc = self.model[0](transformed_image)
                content_loss = self.criterion(transformed_enc[-1], ADaIn_out)

                '''
                Style Loss
                '''
                style_image_style = self.model[0](style)

                style_loss = 0
                for i in range(len(transformed_enc)):
                    style_loss1 = self.model[1].mean_std(transformed_enc[i])
                    style_loss2 = self.model[1].mean_std(style_image_style[i])

                    style_loss += self.criterion(style_loss1[0], style_loss2[0]) + self.criterion(style_loss1[1], style_loss2[1])

                '''Model Update'''
                self.optimizer[0].zero_grad()
                total_gen_loss = content_loss + self.lambda0 * style_loss
                total_gen_loss.backward()
                self.optimizer[0].step()

                last_100_loss.append(total_gen_loss.item())
                running_loss += total_gen_loss.item()
                train_loss.append(total_gen_loss.item())

                if steps % self.print_every == 0:

                    print('\rEpoch {}\tLoss: {:.4f}\n'.format(e, np.mean(last_100_loss)), end="")

                    fpred_image_list.append(util.save_images_to_directory(transformed_image, directory, 'transformed_image_%s.png' % steps))
                    input_image_list.append(util.save_images_to_directory(content, directory, 'content_image_%s.png' % steps))
                    target_image_list.append(util.save_images_to_directory(style, directory, 'style_image_%s.png' % steps))

                    util.raw_score_plotter(train_loss)

            self.save_progress(e, np.mean(last_100_loss))

        self.save_progress(-1, np.mean(last_100_loss))

        image_to_gif('./img/', fpred_image_list, duration=1, gifname='transformed_image_')
        image_to_gif('./img/', target_image_list, duration=1, gifname='style_image_')
        image_to_gif('./img/', input_image_list, duration=1, gifname='content_image_')

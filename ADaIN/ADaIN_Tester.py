
import util
from image_to_gif import image_to_gif
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tester():

    def __init__(self, dataloader, model, checkpoint_path = './model/checkpoint_0.pth'):

        self.dataloader = dataloader
        self.model = model

        self.checkpoint_path = checkpoint_path


    def load_progress(self,):

        checkpoint = torch.load(self.checkpoint_path)

        self.model[2].load_state_dict(checkpoint['model2_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print("Training Loss From Last Session")
        return epoch, loss

    def test(self):
        steps = 0

        # used to make gifs later
        fpred_image_list = []

        target_image_list = []
        input_image_list = []

        directory = './img/test/'

        self.load_progress()

        for content, style in iter(self.dataloader):

            steps += 1

            content = content.to(device)
            style = style.to(device)

            self.model[0].eval()
            self.model[2].eval()

            content = content.to(device)
            style = style.to(device)

            content_enc = self.model[0](content)
            style_enc = self.model[0](style)

            ADaIn_out = self.model[1].ADaIN([content_enc[-1], style_enc[-1]])
            transformed_image = self.model[2](ADaIn_out, content_enc[-1])

            fpred_image_list.append(util.save_images_to_directory(transformed_image, directory, 'transformed_image_%s.png' % steps))
            input_image_list.append(util.save_images_to_directory(content, directory, 'content_image_%s.png' % steps))
            target_image_list.append(util.save_images_to_directory(style, directory, 'style_image_%s.png' % steps))

        image_to_gif(directory, fpred_image_list, duration=1, gifname='transformed_image_')
        image_to_gif(directory, target_image_list, duration=1, gifname='style_image_')
        image_to_gif(directory, input_image_list, duration=1, gifname='content_image_')







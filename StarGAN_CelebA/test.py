
import torch
import os
import torch.nn as nn
import numpy as np

import util
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from image_to_gif import image_to_gif

class Gan_Tester():
    def __init__(self, opts, G, loader, checkpoint):

        self.opts = opts
        self.G = G
        self.loader = loader
        self.checkpoint_path = checkpoint

    def load_progress(self,):

        checkpoint = torch.load(self.checkpoint_path)

        self.G.load_state_dict(checkpoint['Generator_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print("Training Loss From Last Session")
        return epoch, loss

    def test(self):

        _, _ = self.load_progress()

        directory = './img/test/'

        filelist = []

        for index, image in enumerate(self.loader):

            '''Real Images'''
            gen_image = util.stargan_side_by_side_images(self.opts, self.G, image.to(device))
            filelist.append(util.save_images_to_directory(gen_image, directory, 'test_image_s%s.png' % index, nrow=1))

        #create a gif
        image_to_gif(directory, filelist, duration=1.5, gifname='test_images')






import util
from image_to_gif import image_to_gif
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tester():

    def __init__(self, dataloader, model, checkpoint_path = './model/checkpoint_0.pth'):

        self.dataloader = dataloader
        self.model = model

        self.checkpoint_path = checkpoint_path


    def load_progress(self,):

        # TODO get rid of hardcoding and come up with an automated way.

        checkpoint = torch.load(self.checkpoint_path)

        self.model[0].load_state_dict(checkpoint['model0_state_dict'])
        self.model[1].load_state_dict(checkpoint['model1_state_dict'])
        self.model[2].load_state_dict(checkpoint['model2_state_dict'])
        self.model[3].load_state_dict(checkpoint['model3_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print("Training Loss From Last Session")
        return epoch, loss

    def test(self):
        steps = 0

        # used to make gifs later
        fpred_image_list = []
        fgf_reconstruct = []
        gfg_reconstruct = []
        gpred_image_list = []
        target_image_list = []
        input_image_list = []

        directory = './img/test/'

        self.load_progress()

        for input_image, target_image in iter(self.dataloader):

            steps += 1

            input_image = input_image.to(device)
            target_image = target_image.to(device)

            self.model[0].eval()
            self.model[1].eval()
            self.model[2].eval()
            self.model[3].eval()

            mu1, z1 = self.model[0](input_image)
            mu2, z2 = self.model[1](target_image)

            '''L_GAN'''
            '''cross domain'''
            x2_1_recon = self.model[2](z2 + mu2)
            x1_2_recon = self.model[3](z1 + mu1)

            fpred_image_list.append(util.save_images_to_directory(x2_1_recon, directory, 'x2_1_recon_image_%s.png' % steps))
            gpred_image_list.append(util.save_images_to_directory(x1_2_recon, directory, 'x1_2_recon_image_%s.png' % steps))
            input_image_list.append(util.save_images_to_directory(input_image, directory, 'input_image_%s.png' % steps))
            target_image_list.append(util.save_images_to_directory(target_image, directory, 'target_image_%s.png' % steps))

        image_to_gif('./img/test/', fpred_image_list, duration=1, gifname='x2_1_recon')
        image_to_gif('./img/test/', gpred_image_list, duration=1, gifname='x1_2_recon')
        image_to_gif('./img/test/', target_image_list, duration=1, gifname='target')
        image_to_gif('./img/test/', input_image_list, duration=1, gifname='input')







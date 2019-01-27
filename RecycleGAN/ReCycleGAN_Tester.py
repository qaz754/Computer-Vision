
import util
from image_to_gif import image_to_gif
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tester():

    def __init__(self, dataloader, model):

        self.dataloader = dataloader
        self.model = model

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

        for input_image, target_image in iter(self.dataloader):

            steps += 1

            input_image = input_image.to(device)
            target_image = target_image.to(device)

            '''Input(F) --> target(G)'''

            g_pred = self.model[0](input_image)

            '''Input(G) --> target(F)'''

            f_pred = self.model[1](target_image)

            '''F_discriminator'''
            '''F_discriminator loss for the generator F'''

            '''Cycle Consistency'''

            '''F(x) -> G(F(x) -> X'''

            FGF_pred = self.model[1](self.model[0](input_image))
            GFG_pred = self.model[0](self.model[1](target_image))

            '''Discriminators'''
            '''G Discriminator'''
            '''Model Update'''

            fpred_image_list.append(util.save_images_to_directory(f_pred, directory, 'f_pred_image_%s.png' % steps))
            gpred_image_list.append(util.save_images_to_directory(g_pred, directory, 'g_pred_image_%s.png' % steps))
            fgf_reconstruct.append(util.save_images_to_directory(FGF_pred, directory, 'FGF_reconstruct_%s.png' % steps))
            gfg_reconstruct.append(util.save_images_to_directory(GFG_pred, directory, 'GFG_reconstruct_%s.png' % steps))
            input_image_list.append(util.save_images_to_directory(input_image, directory, 'input_image_%s.png' % steps))
            target_image_list.append(util.save_images_to_directory(target_image, directory, 'target_image_%s.png' % steps))


        image_to_gif('./img/test/', fpred_image_list, duration=1, gifname='fpred')
        image_to_gif('./img/test/', gpred_image_list, duration=1, gifname='gpred')
        image_to_gif('./img/test/', target_image_list, duration=1, gifname='target')
        image_to_gif('./img/test/', input_image_list, duration=1, gifname='input')
        image_to_gif('./img/test/', fgf_reconstruct, duration=1, gifname='FGF_Reconstruct')
        image_to_gif('./img/test/', gfg_reconstruct, duration=1, gifname='GFG_Reconstruct')







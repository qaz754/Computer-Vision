#Inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/colorization_dataset.py

import numpy as np
import random

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from image_folder import get_images

import torchvision.transforms.functional as TVF

class AB_Combined_ImageLoader(Dataset):
    """
    Dataloader class used to load in data in an image folder.
    This works so that an image of 2 images A, B that are concatenated horizontally can be split up with the right transformations
    """

    def __init__(self, img_folder, target_folder, transform=None, shuffle=True, train=True, num_images=2, size=256, randomcrop=196, hflip=0.5, vflip=0.5):
        '''

        :param img_folder:
        :param transform:
        :param additional_transform:
        :param final_transformation:
        '''

        self.img_folder = img_folder
        self.size = size
        self.num_images = num_images #for resizing
        self.randomCrop = randomcrop #randomcrop
        self.hflip = hflip
        self.vflip = vflip
        self.transforms = transform
        self.shuffle = shuffle
        self.train = train

        self.file_names = get_images(img_folder)
        self.target_names = get_images(target_folder)

    def __len__(self):

        return len(self.file_names) - self.num_images

    def __getitem__(self, index):

        combined_A = Image.open(self.file_names[index]).convert('RGB')
        combined_B = Image.open(self.target_names[index]).convert('RGB')

        resize = transforms.Resize(size=(self.size, self.num_images * self.size))

        combined_A = resize(combined_A)
        combined_B = resize(combined_B)

        combined_A, combined_B = crop_PIL(combined_A, combined_B, self.num_images, crop_size=self.randomCrop, random=True)

        if self.train:

            if random.random() >= self.hflip:
                for i in range(self.num_images):
                    combined_A[i] = TVF.hflip(combined_A[i])
                    combined_B[i] = TVF.hflip(combined_B[i])

            if random.random() >= self.vflip:
                for i in range(self.num_images):
                    combined_A[i] = TVF.hflip(combined_A[i])
                    combined_B[i] = TVF.hflip(combined_B[i])

        for i in range(self.num_images):
            combined_A[i] = self.transforms(combined_A[i])
            combined_B[i] = self.transforms(combined_B[i])

        left_image = combined_A
        right_image = combined_B

        return left_image, right_image


def crop_PIL(input_image, target_image, num_image, crop_size=0, random=False):

    #assumes channel X Height X Width

    w = input_image.size[0]
    h = input_image.size[1]

    assert w % num_image == 0, "The Width is not a multiple of the number of splits"
    w_cutoff = w // num_image

    w_crop = 0
    h_crop = 0

    if random != False:
        w_crop = np.random.randint(0, w_cutoff - crop_size)
        h_crop = np.random.randint(0, h - crop_size)

    input_image_list = []
    target_image_list = []

    for i in range(num_image):
        starting_point = i * w_cutoff + w_crop
        input_image_list.append(input_image.crop((starting_point, h_crop, crop_size + starting_point, crop_size + h_crop)))
        target_image_list.append(target_image.crop((starting_point, h_crop, crop_size + starting_point, crop_size + h_crop)))

    return input_image_list, target_image_list

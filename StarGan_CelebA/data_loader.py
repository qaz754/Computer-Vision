#Inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/colorization_dataset.py

import numpy as np
import pandas as pd
import random

from torch.utils.data import Dataset
from torchvision import transforms

import torch


from PIL import Image
from skimage import color

from image_folder import get_images

import torchvision.transforms.functional as TVF

attributes = [
    'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Bangs', 'Straight_Hair', 'Wavy_Hair', 'Young', 'Male'
]

hair_color_attributes = [
    'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair',
]

hair_attributes = [
    'Straight_Hair', 'Wavy_Hair'
]

bang = [
    'Bangs'
]

young = [
    'Young'
]

gender = [
    'Male'
]

target_attributes = [hair_color_attributes, hair_attributes, bang, young, gender]

class Pix2Pix_AB_Dataloader(Dataset):
    """
    Dataloader class used to load in data in an image folder.
    Made it so that it performs a fixed set of transformations to a pair of images in different folders
    """

    def __init__(self, img_folder, attribute_file, target_attributes=target_attributes,  transform=None, size =256, randomcrop = 224, hflip=0.5, vflip=0.5, train=True):
        '''
        :param img_folder:
        :param transform:
        :param additional_transform:
        :param final_transformation:
        '''

        self.img_folder = img_folder
        self.attribute_file = attribute_file
        self.attributes = {}

        self.target_attributes=target_attributes

        self.load_attribute()

        self.size = size #for resizing
        self.randomCrop = randomcrop #randomcrop
        self.hflip = hflip
        self.vflip = vflip
        self.transforms = transform
        self.train = train

        self.file_names = get_images(img_folder)

        #load_attributes
        self.load_attribute()

    def load_attribute(self):
        """
        loads attributes from self.attribute_file using the column as a key to a dictionary and rest of the columns
        as values.
        """

        lines = [line.rstrip() for line in open(self.attribute_file, 'r')]
        self.all_attr_names = lines[1].split()

        for i in range(2 , len(lines)):
            line_split = lines[i].split()
            self.attributes[line_split[0]] = line_split[1:]

    def attribute_chooser(self, attribute_list):

        binary = np.ones(len(attribute_list), dtype=int) * -1
        if len(attribute_list) > 1:
            choice = np.random.randint(0, len(attribute_list))
        else:
            if np.random.rand() <= 0.5:
                choice = 0
            else:
                return attribute_list[0], binary

        chosen = attribute_list[choice]
        chosen_idx = attribute_list.index(chosen)

        binary[chosen_idx] = 1

        return chosen, binary

    def get_transformed_attributes(self, image_name):

        target_attributes = self.attributes[image_name]

        for target_set in self.target_attributes:
            chosen_attribute, binary = self.attribute_chooser(target_set)

            for idx, attr in enumerate(target_set):
                index = self.all_attr_names.index(attr)
                target_attributes[index] = binary[idx]

        return target_attributes

    def list_to_tensor(self, input):
        input = list(map(int, input))
        input = torch.from_numpy(np.asarray(input))

        return input


    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, index):

        left_image = Image.open(self.file_names[index]).convert('RGB')
        image_name = self.file_names[index].replace(self.img_folder, "")

        orig_attributes = self.attributes[image_name]
        transformed_attributes = self.get_transformed_attributes(image_name)

        '''
        Resize
        '''

        resize = transforms.Resize(size=(self.size, self.size))

        left_image = resize(left_image)

        '''
        RandomCrop
        '''
        if self.train:
            i, j, h, w = transforms.RandomCrop.get_params(
                left_image, output_size=(self.randomCrop, self.randomCrop)
            )
            left_image = TVF.crop(left_image, i, j, h, w)

            if random.random() >= self.hflip:
                left_image = TVF.hflip(left_image)

        left_image = self.transforms(left_image)


        return left_image, self.list_to_tensor(orig_attributes), self.list_to_tensor(transformed_attributes)

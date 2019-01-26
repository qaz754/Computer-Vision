#Inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/colorization_dataset.py

from torch.utils.data import Dataset
from torchvision import transforms

from image_folder import get_images

import torchvision.transforms.functional as TVF

from PIL import Image, ImageFilter

class USPS_ImageLoader(Dataset):
    """
    Dataloader class used to load in data in an image folder.
    Takes in USPS data (A), and creates images that are the edges of A
    """

    def __init__(self, img_folder, transform=None, shuffle=True, train=True, num_images=2, size=256, randomcrop=196, hflip=0.5, vflip=0.5):
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
        #self.target_names = get_images(target_folder)

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, index):

        #TODO: Add some randomization to shuffle paried data.

        image = Image.open(self.file_names[index]).convert('RGB')

        resize = transforms.Resize(size=(self.size, self.size))
        left_image = resize(image)
        right_image = left_image.filter(ImageFilter.FIND_EDGES)

        #flip the image
        #right_image = TVF.vflip(right_image)

        left_image = self.transforms(left_image)
        right_image = self.transforms(right_image)

        return left_image, right_image

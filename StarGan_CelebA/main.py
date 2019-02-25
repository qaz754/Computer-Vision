
from data_loader import Pix2Pix_AB_Dataloader
from image_folder import get_folders

import torch
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_dir = '/home/youngwook/Downloads/img_align_celeba/'
attribute_file = '/home/youngwook/Downloads/list_attr_celeba.txt'
folder_names = get_folders(image_dir)

train_folder = folder_names[0]

from torchvision.transforms import transforms
# Define a transform to pre-process the training images.

num_images = 1
size = 144
randomCrop = 128

transform_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
    ])

train_data = Pix2Pix_AB_Dataloader(train_folder, attribute_file=attribute_file, transform=transform_1, size=size, randomcrop=randomCrop)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=12)

from network import discriminator, ResNet
from util import get_optimizer, LS_discriminator_loss, LS_generator_loss
from train import run_vanilla_gan

NOISE_DIM=96
NUM_CLASSES = 40

'''Discriminator'''
D = discriminator(3, NUM_CLASSES).to(device)

'''Generator'''
G = ResNet(3 + NUM_CLASSES, 3, 32).to(device)

'''Optimizers'''
D_solver = get_optimizer(D)
G_solver = get_optimizer(G)

'''run training'''
run_vanilla_gan(D, G, D_solver, G_solver, LS_discriminator_loss, LS_generator_loss, train_loader, num_epochs=5, n_classes=NUM_CLASSES)

'''Generate New Samples'''

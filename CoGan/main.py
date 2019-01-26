
from data_loader import USPS_ImageLoader
from image_folder import get_images, get_folders

import torch
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_dir = 'usps_data/Numerals/'
folder_names = get_folders(image_dir)

train_folder = folder_names[0]
target_folder = folder_names[2]

from torchvision.transforms import transforms
# Define a transform to pre-process the training images.

resize = 28
batch_size = 64

transform_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
    ])

train_data = USPS_ImageLoader(train_folder, transform=transform_1, size=resize)

train_loader = DataLoader(train_data, batch_size=batch_size,
                        shuffle=True, num_workers=4)


from network import discriminator, generator
from util import LS_discriminator_loss, LS_generator_loss
from train import run_vanilla_gan


NOISE_DIM=96

LR = 0.0001
BETA0 = 0.5
BETA1 = 0.999

a, b = next(iter(train_loader))

'''Discriminator'''
D = discriminator(train_loader.batch_size).to(device)

'''Generator'''
G = generator(NOISE_DIM, a.shape[1:4]).to(device)

'''Optimizers'''
D_solver = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA0, BETA1))
G_solver = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA0, BETA1))

'''run training'''
run_vanilla_gan(D, G, D_solver, G_solver, LS_discriminator_loss, LS_generator_loss, train_loader, batch_size=batch_size, num_epochs=200)

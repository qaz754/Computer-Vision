
from data_loader import AB_Combined_ImageLoader
from image_folder import get_images, get_folders

import os
import torch
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn as nn
#from augmentation import Rescale, RandomCrop, Normalize, ToTensor, RotateScale, HorizontalFlip, VerticalFlip

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_dir = '/home/youngwook/Downloads/edges2shoes'
folder_names = get_folders(image_dir)

train_folder = folder_names[2]
val_folder = folder_names[1]


from torchvision.transforms import transforms
# Define a transform to pre-process the training images.

num_images = 2
size = 256
randomCrop = 224

transform_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
    ])

train_data = AB_Combined_ImageLoader(train_folder, transform=transform_1, size=size, num_images=num_images, randomcrop=randomCrop)

train_loader = DataLoader(train_data, batch_size=12,
                        shuffle=True, num_workers=4)

test_data = AB_Combined_ImageLoader(val_folder, transform=transform_1, size=randomCrop, num_images=num_images, train=False, randomcrop=randomCrop, hflip=0, vflip=0)

test_loader = DataLoader(test_data, batch_size=12,
                        shuffle=True, num_workers=4)

from network import AutoEncoder_Unet, MultiScaleDiscriminator, Encoder


from combined_trainer import trainer

AE = AutoEncoder_Unet(11, 3).to(device)
D_cVAE = MultiScaleDiscriminator(3, 2).to(device)
D_cLR = MultiScaleDiscriminator(3, 2).to(device)
E = Encoder(8).to(device)
LR = 0.0002

criterion = nn.L1Loss()

unet_optim = optim.Adam(AE.parameters(), lr=LR, betas=(0.5, 0.999))
D_cVAE_optim = optim.Adam(D_cVAE.parameters(), lr=LR, betas=(0.5, 0.999))
D_cLR_optim = optim.Adam(D_cLR.parameters(), lr=LR, betas=(0.5, 0.999))
E_optim = optim.Adam(E.parameters(), lr=LR, betas=(0.5, 0.999))

optim_list = [unet_optim, D_cVAE_optim, D_cLR_optim, E_optim]
model_list = [AE, D_cVAE, D_cLR, E]

epochs = 30


trainer = trainer(epochs, train_loader, model_list, optim_list, criterion, resume=False, lambd_l1=10, lamb_kl=0.01, lamb_latent=0.5)

#trains the model
trainer.train()

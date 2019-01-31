
from data_loader import AB_Combined_ImageLoader
from image_folder import get_folders

import torch
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn as nn
from UNIT_Tester import Tester

from network import Encoder, discriminator, ResBlock, Decoder
from UNIT_Trainer import trainer

import itertools


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_dir = '/home/youngwook/Downloads/edges2shoes'
folder_names = get_folders(image_dir)

train_folder = folder_names[2]
val_folder = folder_names[1]

from torchvision.transforms import transforms
# Define a transform to pre-process the training images.

num_images = 2
size = 224
randomCrop = 196

transform_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
    ])

train_data = AB_Combined_ImageLoader(train_folder, transform=transform_1, size=size, num_images=num_images, randomcrop=randomCrop)

train_loader = DataLoader(train_data, batch_size=1,
                        shuffle=True, num_workers=4)

test_data = AB_Combined_ImageLoader(val_folder, transform=transform_1, size=randomCrop, num_images=num_images, train=False, randomcrop=randomCrop, hflip=0, vflip=0)

test_loader = DataLoader(test_data, batch_size=1,
                        shuffle=True, num_workers=4)

Shared_Encoder = ResBlock(128).to(device)
Enc1 = Encoder(3, 3, 32, Shared_Encoder).to(device)
Enc2 = Encoder(3, 3, 32, Shared_Encoder).to(device)

Shared_Decoder = ResBlock(128).to(device)
Dec1 = Decoder(128, 3, 32, Shared_Decoder).to(device)
Dec2 = Decoder(128, 3, 32, Shared_Decoder).to(device)

Disc1 = discriminator(3, 1).to(device)
Disc2 = discriminator(3, 1).to(device)

LR = 0.0001

criterion = nn.L1Loss()

gen_optim = optim.Adam(itertools.chain(Enc1.parameters(), Enc2.parameters(), Dec1.parameters(), Dec2.parameters()), lr=LR, betas=(0.5, 0.999))

G_D_optim = optim.Adam(Disc1.parameters(), lr=LR, betas=(0.5, 0.999))
F_D_optim = optim.Adam(Disc2.parameters(), lr=LR, betas=(0.5, 0.999))

optim_list = [gen_optim, G_D_optim, F_D_optim]
model_list = [Enc1, Enc2, Dec1, Dec2, Disc1, Disc2]

epochs = 5

trainer = trainer(epochs, train_loader, model_list, optim_list, criterion, resume=False)

#trains the model
#trainer.train()

model_list = [Enc1, Enc2, Dec1, Dec2, Disc1, Disc2]

tester = Tester(test_loader, model_list,  checkpoint_path= './model/checkpoint_4.pth')
tester.test()

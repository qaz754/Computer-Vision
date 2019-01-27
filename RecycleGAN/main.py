
from data_loader import AB_Combined_ImageLoader
from image_folder import get_folders

import torch
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn as nn

import itertools
from ReCycleGAN_Tester import Tester

from network import ResNet, discriminator
from Unet import AutoEncoder_Unet
from ReCycleGAN_Trainer import trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_dir = '/home/youngwook/Downloads/Viper/data/recycle-gan'
folder_names = get_folders(image_dir)

Folder_A = folder_names[2]
Folder_B = folder_names[1]

from torchvision.transforms import transforms
# Define a transform to pre-process the training images.

num_images = 3
size = 256
randomCrop = 196

transform_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
    ])

train_data = AB_Combined_ImageLoader(Folder_A, Folder_B, transform=transform_1, size=size, num_images=num_images, randomcrop=randomCrop)

train_loader = DataLoader(train_data, batch_size=1,
                        shuffle=True, num_workers=4)

test_data = AB_Combined_ImageLoader(Folder_A, Folder_B, transform=transform_1, size=randomCrop, num_images=num_images, train=False, randomcrop=randomCrop, hflip=0, vflip=0)

test_loader = DataLoader(test_data, batch_size=1,
                        shuffle=True, num_workers=4)

horizon = num_images - 1

Input_Generator = ResNet(3, 3, 32).to(device)
Target_Generator = ResNet(3, 3, 32).to(device)

Input_Discriminator = discriminator(3, 1).to(device)
Target_Discriminator = discriminator(3, 1).to(device)

Input_Image_Predictor = AutoEncoder_Unet(3 * horizon, 3).to(device)
Target_Image_Predictor = AutoEncoder_Unet(3 * horizon, 3).to(device)

LR = 0.0002

criterion = nn.L1Loss()

gen_optim = optim.Adam(itertools.chain(Input_Generator.parameters(), Target_Generator.parameters(), Input_Image_Predictor.parameters(), Target_Image_Predictor.parameters()), lr=LR, betas=(0.5, 0.999))

Target_Discriminator_optim = optim.Adam(Target_Discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
Input_Discriminator_optim = optim.Adam(Input_Discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

optim_list = [gen_optim, Target_Discriminator_optim, Input_Discriminator_optim]
model_list = [Target_Generator, Input_Generator, Target_Discriminator, Input_Discriminator, Target_Image_Predictor, Input_Image_Predictor]

epochs = 15

trainer = trainer(epochs, train_loader, model_list, optim_list, criterion, cycle_lambda=10, pred_lambda=10)

#trains the model
trainer.train()


#Test the model
'''
F = ResNet(3, 3, 32).to(device)
G = ResNet(3, 3, 32).to(device)

state_dict = torch.load('./model/F_Gen.pth')
F.load_state_dict(state_dict)

state_dict = torch.load('./model/G_Gen.pth')
G.load_state_dict(state_dict)

model_list = [F, G]

tester = Tester(test_loader, model_list)
tester.test()
'''
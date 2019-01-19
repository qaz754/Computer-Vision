
from data_loader import AB_Combined_ImageLoader
from image_folder import get_folders

import torch
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn as nn
from CycleGAN_Tester import Tester

from network import ResNet, discriminator
from CycleGAN_Trainer import trainer

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


F = ResNet(3, 3, 32).to(device)
G = ResNet(3, 3, 32).to(device)

G_D = discriminator(3, 1).to(device)
F_D = discriminator(3, 1).to(device)

LR = 0.0002

criterion = nn.L1Loss()

gen_optim = optim.Adam(itertools.chain(F.parameters(), G.parameters()), lr=LR, betas=(0.5, 0.999))

G_D_optim = optim.Adam(G_D.parameters(), lr=LR, betas=(0.5, 0.999))
F_D_optim = optim.Adam(F_D.parameters(), lr=LR, betas=(0.5, 0.999))


state_dict = torch.load('./model/F_Gen_49825.pth')
F.load_state_dict(state_dict)

state_dict = torch.load('./model/G_Gen_49825.pth')
G.load_state_dict(state_dict)

state_dict = torch.load('./model/F_Discrim_49825.pth')
F_D.load_state_dict(state_dict)

state_dict = torch.load('./model/G_Discrim_49825.pth')
G_D.load_state_dict(state_dict)




optim_list = [gen_optim, G_D_optim, F_D_optim]
model_list = [F, G, G_D, F_D]

epochs = 1

trainer = trainer(epochs, train_loader, model_list, optim_list, criterion, cycle_lambda=10, resume=True)

#trains the model
trainer.train()


#Test the model
#F = ResNet(3, 3, 32).to(device)
#G = ResNet(3, 3, 32).to(device)

model_list = [F, G]

tester = Tester(test_loader, model_list)
tester.test()

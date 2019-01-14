
from data_loader import Pix2Pix_AB_Dataloader
from image_folder import get_folders

import torch
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_dir = '/home/youngwook/Downloads/Carvana'
folder_names = get_folders(image_dir)

train_folder = folder_names[1]
target_folder = folder_names[2]

from torchvision.transforms import transforms
# Define a transform to pre-process the training images.

resize = 256
randomCrop = 224

transform_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
    ])

train_data = Pix2Pix_AB_Dataloader(train_folder, target_folder, transform=transform_1, size = resize, randomcrop=randomCrop)

#train_data = Pix2Pix_Dataloader(train_folder, transform=transform_train, additional_transform=GrayScaleAndColor1)

train_loader = DataLoader(train_data, batch_size=1,
                        shuffle=True, num_workers=4)

from skimage import color

from network import ResNet, discriminator
from CycleGAN_Trainer import trainer

F = ResNet(3, 3, 32).to(device)
G = ResNet(3, 3, 32).to(device)

G_D = discriminator(3, 1).to(device)
F_D = discriminator(3, 1).to(device)

LR = 0.0002

criterion = nn.L1Loss()

gen_params = list(F.parameters()) + list(G.parameters())
gen_optim = optim.Adam(gen_params, lr=LR, betas=(0.5, 0.999))
G_D_optim = optim.Adam(G_D.parameters(), lr=LR, betas=(0.5, 0.999))
F_D_optim = optim.Adam(F_D.parameters(), lr=LR, betas=(0.5, 0.999))

optim_list = [gen_optim, G_D_optim, F_D_optim]
model_list = [F, G, G_D, F_D]

epochs = 1

trainer = trainer(epochs, train_loader, model_list, optim_list, criterion, cycle_lambda=10)

#trains the model
trainer.train()


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

#options
from options import options
options = options()
opts = options.parse()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torchvision.transforms import transforms
# Define a transform to pre-process the training images.

num_images = opts.num_images
size = opts.resize
randomCrop = opts.image_shape

transform_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
    ])

train_data = AB_Combined_ImageLoader(opts.train_folder, transform=transform_1, size=size, num_images=num_images, randomcrop=randomCrop)
train_loader = DataLoader(train_data, batch_size=opts.batch, shuffle=True, num_workers=opts.cpu_count)

test_data = AB_Combined_ImageLoader(opts.val_folder, transform=transform_1, size=randomCrop, num_images=num_images, train=False, randomcrop=randomCrop, hflip=0, vflip=0)
test_loader = DataLoader(test_data, batch_size=opts.batch, shuffle=True, num_workers=opts.cpu_count)

F = ResNet(opts).to(device)
G = ResNet(opts).to(device)

G_D = discriminator(opts).to(device)
F_D = discriminator(opts).to(device)

LR = opts.lr

if opts.criterion == 'l1':
    criterion = nn.L1Loss()
elif opts.criterion == 'l2':
    criterion = nn.MSELoss()

gen_optim = optim.Adam(itertools.chain(F.parameters(), G.parameters()), lr=LR, betas=(opts.beta1, opts.beta2))

G_D_optim = optim.Adam(G_D.parameters(), lr=LR, betas=(opts.beta1, opts.beta2))
F_D_optim = optim.Adam(F_D.parameters(), lr=LR, betas=(opts.beta1, opts.beta2))

optim_list = [gen_optim, G_D_optim, F_D_optim]
model_list = [F, G, G_D, F_D]

if opts.print_model:
    for i in model_list:
        print(i)

trainer = trainer(opts, train_loader, model_list, optim_list, criterion)

#trains the model
trainer.train()

model_list = [F, G]

tester = Tester(test_loader, model_list, checkpoint_path= './model/checkpoint_0.pth')
tester.test()

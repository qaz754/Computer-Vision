import torch
import data_loader
import image_folder

import test_loader

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_dir = '/home/youngwook/Downloads/img_align_celeba/'
attribute_file = '/home/youngwook/Downloads/list_attr_celeba.txt'
folder_names = image_folder.get_folders(image_dir)

test_dir = '/home/youngwook/Downloads/faces/'
test_folder = image_folder.get_folders(test_dir)


#options
from options import options
options = options()
opts = options.parse()

#Download and load the training data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

train_data = data_loader.CelebA_DataLoader(folder_names[0], transform=transform, attribute_file=attribute_file, size=opts.resize, randomcrop=opts.image_shape)
trainloader = DataLoader(train_data, batch_size=opts.batch, shuffle=True, num_workers=4)

test_data = test_loader.CelebA_DataLoader(test_folder[0], transform=transform, size=opts.image_shape)
testloader = DataLoader(test_data, batch_size=1)

from network import discriminator, ResNet
from train import GAN_Trainer
from test import Gan_Tester

'''Discriminator'''
D = discriminator(opts).to(device)

'''Generator'''
G = ResNet(opts).to(device)

'''Optimizers'''
import torch.optim as optim

G_optim = optim.Adam(G.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))
D_optim = optim.Adam(D.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))

'''print the model'''
if opts.print_model:
    print(G)
    print(D)

'''run training'''
#trainer = GAN_Trainer(opts, D, G, D_optim, G_optim, trainloader)
#trainer.train()

tester = Gan_Tester(opts, G, testloader, checkpoint='./model/checkpoint_-1.pth')
tester.test()
'''Generate New Samples'''

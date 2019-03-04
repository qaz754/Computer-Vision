import torch
import data_loader
import image_folder

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_dir = '/home/youngwook/Downloads/img_align_celeba/'
attribute_file = '/home/youngwook/Downloads/list_attr_celeba.txt'
folder_names = image_folder.get_folders(image_dir)


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
trainloader = DataLoader(train_data, batch_size=opts.batch, shuffle=True, num_workers=12)

from network import discriminator, ResNet
from train import GAN_Trainer

'''Discriminator'''
D = discriminator(opts).to(device)

'''Generator'''
G = ResNet(opts).to(device)

'''Optimizers'''
import torch.optim as optim

G_optim = optim.Adam(G.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))
D_optim = optim.Adam(D.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))

'''run training'''
trainer = GAN_Trainer(opts, D, G, D_optim, G_optim, trainloader)
trainer.train()

'''Generate New Samples'''

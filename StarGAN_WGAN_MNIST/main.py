import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
    ])
#options
from options import options
options = options()
opts = options.parse()

#Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train= True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opts.batch, shuffle=True)

testset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=opts.batch, shuffle=False)

from network import discriminator, ResNet
from train import run_vanilla_gan

'''Discriminator'''
D = discriminator(opts).to(device)

'''Generator'''
G = ResNet(opts).to(device)

'''Optimizers'''
import torch.optim as optim

G_optim = optim.Adam(G.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))
D_optim = optim.Adam(D.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))

'''run training'''
run_vanilla_gan(opts, D, G, D_optim, G_optim, trainloader)

'''Generate New Samples'''

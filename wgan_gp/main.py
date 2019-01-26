
import torch

#load mnist dataset and define network
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor()])

#Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train= True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = datasets.MNIST('MNIST_data/', download=False, train=False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle=False)

from network import discriminator, generator
from util import LS_discriminator_loss, LS_generator_loss
from train import run_vanilla_gan


NOISE_DIM=96

LR = 0.0001
BETA0 = 0.5
BETA1 = 0.9
x, _ = next(iter(trainloader))

'''Discriminator'''
D = discriminator(x.shape[1:4]).to(device)

'''Generator'''
G = generator(NOISE_DIM, x.shape[1:4]).to(device)

'''Optimizers'''
D_solver = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA0, BETA1))
G_solver = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA0, BETA1))

'''run training'''
run_vanilla_gan(D, G, D_solver, G_solver, LS_discriminator_loss, LS_generator_loss, trainloader, num_epochs=200, n_critic=5, clip_value=0.01)

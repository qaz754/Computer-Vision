
import torch
from torchvision import datasets, transforms

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


#Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor()])

#Download and load the training data
batch_size = 128

trainset = datasets.MNIST('MNIST_data/', download=True, train= True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = datasets.MNIST('MNIST_data/', download=False, train=False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle=False)

from network import discriminator, generator
from util import get_optimizer, discriminator_loss, generator_loss
from train import run_vanilla_gan

NOISE_DIM=96

'''Discriminator'''
D = discriminator(batch_size).to(device)

'''Generator'''
G = generator(batch_size, NOISE_DIM).to(device)

'''Optimizers'''
D_solver = get_optimizer(D)
G_solver = get_optimizer(G)

'''run training'''
run_vanilla_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, trainloader, num_epochs=15)

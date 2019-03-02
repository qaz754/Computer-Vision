
import torch

#load mnist dataset and define network
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#options
from options import options
options = options()
opts = options.parse()

#Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
#Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train= True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opts.batch, shuffle=True)

testset = datasets.MNIST('MNIST_data/', download=False, train=False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = opts.batch, shuffle=False)

from network import discriminator, generator
from train import run_vanilla_gan

LR = opts.lr
BETA0 = opts.beta1
BETA1 = opts.beta2
x, _ = next(iter(trainloader))

'''Discriminator'''
D = discriminator(opts).to(device)

'''Generator'''
G = generator(opts).to(device)

if opts.print_model:
    print(D)
    print(G)

'''Optimizers'''
D_solver = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA0, BETA1))
G_solver = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA0, BETA1))

'''run training'''
run_vanilla_gan(opts, D, G, D_solver, G_solver, trainloader)

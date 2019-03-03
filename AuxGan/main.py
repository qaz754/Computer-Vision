
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

'''Generate New Samples'''

from util import categorical_label_generator, one_hot_encoder, sample_noise, show_images
from image_to_gif import image_to_gif
import os
import matplotlib.pyplot as plt
import numpy as np

batch_size = opts.batch
filelist = []


for i in range(0, opts.num_classes):

    #generate target labels
    label_array = np.full((batch_size, 1), i)
    label_array = one_hot_encoder(label_array, n_classes=opts.num_classes)
    label_array = torch.from_numpy(label_array).float().to(device)

    #generate noise
    g_fake_seed = sample_noise(batch_size, opts.noise_dim).to(device)
    g_fake_seed = torch.cat((g_fake_seed, label_array), dim=1)

    #change the network to eval mode
    G.eval()
    fake_images = G(g_fake_seed)

    #show and save the image.
    imgs_numpy = fake_images.data.cpu().numpy()

    '''filename used for saving the image'''
    directory = './img/'
    filename = 'Generated_with_Label_%s.png' % i
    filelist.append(filename)

    filename = os.path.join('%s' % directory, '%s' % filename)

    show_images(imgs_numpy[:opts.batch], filename, i, title='AuxGANs With Label %s')
    plt.show()
    print()

image_to_gif('./img/', filelist, duration=1, gifname='movie2')


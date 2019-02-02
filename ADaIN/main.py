
from data_loader import Pix2Pix_AB_Dataloader
from image_folder import get_folders

import torch
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn as nn
from ADaIN_Tester import Tester

from network import AdaIN, Decoder
from ADaIN_Trainer import trainer

#import encoder (a custom vgg19)
import vgg19

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_dir = '/home/youngwook/cocoapi/images'
folder_names = get_folders(image_dir)

content_folder = folder_names[1]
val_folder = folder_names[2]

image_dir = '/home/youngwook/Downloads/paint'
folder_names = get_folders(image_dir)

style_folder = folder_names[1]


from torchvision.transforms import transforms
# Define a transform to pre-process the training images.

size = 512
randomCrop = 256

transform_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
    ])

train_data = Pix2Pix_AB_Dataloader(content_folder, style_folder, transform=transform_1, size=size, randomcrop=randomCrop, hflip=1, vflip=1)
train_loader = DataLoader(train_data, batch_size=8,shuffle=True, num_workers=5)

AdaIN = AdaIN()

encoder = vgg19.normalized_vgg
encoder.load_state_dict(torch.load('model/vgg_normalised.pth'))
encoder = vgg19.VGG_normalized(encoder)
encoder.to(device)

Dec = Decoder(vgg19.vgg19(pretrained=False)).to(device)

LR = 0.0001

criterion = nn.MSELoss()

Decoder_optim = optim.Adam(Dec.parameters(), lr=LR, betas=(0.5, 0.999))
model_list = [encoder, AdaIN, Dec]
optim_list = [Decoder_optim]

epochs = 10

trainer = trainer(epochs, train_loader, model_list, optim_list, criterion, resume=False)

#trains the model
#trainer.train()


test_data = Pix2Pix_AB_Dataloader(val_folder, style_folder, transform=transform_1, size=randomCrop, train=False, randomcrop=randomCrop, hflip=0, vflip=0)
test_loader = DataLoader(test_data, batch_size=8, shuffle=True, num_workers=4)


tester = Tester(test_loader, model_list,  checkpoint_path= './model/checkpoint_-1.pth')
tester.test()


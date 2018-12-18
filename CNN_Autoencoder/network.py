

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AE(nn.Module):
    
    def __init__(self):
        super(AE, self).__init__()



        #define the layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1), #batch_size * 16 * 10 * 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),                #batch_size * 16 * 5 * 5
            nn.Conv2d(16,8,3, stride=2, padding=2),   #batch_size * 8 * 3 * 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)                 #batch_size * 8 * 2 * 2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  #batch_size * 16 * 5 * 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride= 3, padding=1), #batch_size * 8 * 15 * 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  #batch_size * 1 * 28 * 28
            nn.Tanh()
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x



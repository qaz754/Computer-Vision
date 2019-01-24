

import torch
import torch.nn as nn
from util import Flatten, spatially_replicate

from torch.autograd import Variable

from torchvision.models import resnet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def if_tensor(tensor):

    if isinstance(tensor, torch.Tensor):
        return True

class conv_down(nn.Module):

    def __init__(self, channel_input, channel_output, kernel=3, stride=1, padding=0, Norm=True, Dropout=0.0):
        super(conv_down, self).__init__()

        steps = [nn.Conv2d(channel_input, channel_output, kernel_size=kernel, stride=stride, padding=padding, bias=False)]

        if Norm:
            steps.append(nn.BatchNorm2d(channel_output))
        steps.append(nn.LeakyReLU(0.2))

        if Dropout > 0:
            steps.append(nn.Dropout(Dropout))

        self.model = nn.Sequential(*steps)

    def forward(self, x):

        return self.model(x)

class conv_up(nn.Module):

    def __init__(self, channel_input, channel_output, kernel=3, stride=1, padding=0, Norm=True, Dropout=0.0):
        super(conv_up, self).__init__()

        steps = [nn.ConvTranspose2d(channel_input, channel_output, kernel_size=kernel, stride=stride, padding=padding, bias=False)]

        if Norm:
            steps.append(nn.BatchNorm2d(channel_output))
        steps.append(nn.ReLU(inplace=True))

        if Dropout > 0:
            steps.append(nn.Dropout(Dropout))

        self.model = nn.Sequential(*steps)

    def forward(self, x, skip_input=None):

        if if_tensor(skip_input):
            x = torch.cat((x, skip_input), 1)

        return self.model(x)

class AutoEncoder_Unet(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(AutoEncoder_Unet, self).__init__()

        self.U_down1 = conv_down(channel_input=in_channel, channel_output=16, kernel=4, stride=2, padding=1, Norm=False)
        self.U_down2 = conv_down(channel_input=16, channel_output=32, kernel=4, stride=2, padding=1)
        self.U_down3 = conv_down(channel_input=32, channel_output=64, kernel=4, stride=2, padding=1)
        self.U_down4 = conv_down(channel_input=64, channel_output=128, kernel=4, stride=2, padding=1)
        self.U_down5 = conv_down(channel_input=128, channel_output=256, kernel=4, stride=2, padding=1)
        self.U_down6 = conv_down(channel_input=256, channel_output=512, kernel=4, stride=1, padding=1)
        self.U_down7 = conv_down(channel_input=512, channel_output=512, kernel=4, stride=1, padding=1)
        self.U_down8 = conv_down(channel_input=512, channel_output=1024, kernel=4, stride=1, padding=1)
        self.U_down9 = conv_down(channel_input=1024, channel_output=1024, kernel=4, stride=1, padding=1,  Norm=False)

        self.U_up1 = conv_up(channel_input=1024, channel_output=1024, kernel=2, stride=1)
        self.U_up2 = conv_up(channel_input=2048, channel_output=512, kernel=2, stride=1)
        self.U_up3 = conv_up(channel_input=1024, channel_output=512, kernel=2, stride=1)
        self.U_up4 = conv_up(channel_input=1024, channel_output=256, kernel=2, stride=1)
        self.U_up5 = conv_up(channel_input=512, channel_output=128, kernel=2, stride=2)
        self.U_up6 = conv_up(channel_input=256, channel_output=64, kernel=2, stride=2)
        self.U_up7 = conv_up(channel_input=128, channel_output=32, kernel=2, stride=2)
        self.U_up8 = conv_up(channel_input=64, channel_output=16, kernel=2, stride=2)
        self.U_up9 = conv_up(channel_input=32, channel_output=out_channel, kernel=2, stride=2)

        self.tan_output = nn.Sequential(
            nn.Tanh()
        )
    def forward(self, x, z_code):

        z_code = spatially_replicate(z_code, x.shape[3])

        '''Implementation where the noise is injected to the input only'''
        x = torch.cat((x, z_code), 1)

        input_down1 = self.U_down1(x)
        input_down2 = self.U_down2(input_down1)
        input_down3 = self.U_down3(input_down2)
        input_down4 = self.U_down4(input_down3)
        input_down5 = self.U_down5(input_down4)
        input_down6 = self.U_down6(input_down5)
        input_down7 = self.U_down7(input_down6)
        input_down8 = self.U_down8(input_down7)
        input_down9 = self.U_down9(input_down8)

        x = self.U_up1(input_down9)
        x = self.U_up2(x, input_down8)
        x = self.U_up3(x, input_down7)
        x = self.U_up4(x, input_down6)
        x = self.U_up5(x, input_down5)
        x = self.U_up6(x, input_down4)
        x = self.U_up7(x, input_down3)
        x = self.U_up8(x, input_down2)
        x = self.U_up9(x, input_down1)


        return self.tan_output(x)

'''
Discriminator 
'''


class MultiScaleDiscriminator(nn.Module):
    '''
    PatchDiscriminator class with discriminator at different scales
    '''

    def __init__(self, in_channel, num_discrim=2):
        super(MultiScaleDiscriminator, self).__init__()

        def DiscrimBlock(channel_input, channel_output, kernel=4, stride=2, padding=1, Norm=True, Dropout=0.0):
            '''
            Creates a block for the discriminator
            :param channel_input:
            :param channel_output:
            :param kernel:
            :param stride:
            :param padding:
            :param Norm:
            :param Dropout:
            :return:
            '''

            steps = [nn.Conv2d(channel_input, channel_output, kernel_size=kernel, stride=stride, padding=padding)]
            if Norm:
                steps.append(nn.InstanceNorm2d(channel_output))
            steps.append(nn.LeakyReLU(0.2))
            if Dropout > 0:
                steps.append(nn.Dropout(Dropout))

            return steps

        self.models = nn.ModuleList()
        '''Pytorch list that one can use to pass layers to and build models on the fly'''
        '''https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463'''

        for i in range(num_discrim):
            self.models.add_module('discrim_%d' %i,
                                   nn.Sequential(
                                       *DiscrimBlock(in_channel, 64, Norm=False),
                                       *DiscrimBlock(64, 128),
                                       *DiscrimBlock(128,256),
                                       *DiscrimBlock(256,512),
                                       nn.Conv2d(512, 1, 4, padding=1)
                                   ))

        self.downsample = nn.AvgPool2d(in_channel, stride=2, padding=[1,1], count_include_pad=False)

    def forward(self, x):
        '''
        Takes in an input x and feeds it to all the discriminators with different scales
        :param x:
        :return:
        '''

        output = []

        for i in self.models:
            output.append(i(x))
            x = self.downsample(x)

        return output

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        resnet18_model = resnet18(pretrained=True)

        # Extracts features at the last fully-connected
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)

        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):

        x = self.feature_extractor(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        #randomly sample epsilon

        #esp = (torch.randn(noise_size[0], 16, noise_size[2], noise_size[3]))
        esp = (torch.randn(logvar.size())).to(device)
        #get the reparameterized function

        reparam = mu + logvar.mul(1/2).exp_() * Variable(esp.to(device), requires_grad=False)

        return mu, logvar, reparam


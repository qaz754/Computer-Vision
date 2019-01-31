

#input is 224

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def if_tensor(tensor):

    if isinstance(tensor, torch.Tensor):
        return True

class ResBlock(nn.Module):

    def __init__(self, input_channel):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_channel, input_channel, 3),
            nn.InstanceNorm2d(input_channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_channel, input_channel, 3),
            nn.InstanceNorm2d(input_channel)
        )

    def forward(self, x):

        x = x + self.block(x)

        return x

class Encoder(nn.Module):
    def __init__(self, channel_in, channel_out, up_channel, shared_layer):
        super(Encoder, self).__init__()

        self.shared_layer = shared_layer

        model = nn.Sequential(
                    nn.ReflectionPad2d(3),
                     nn.Conv2d(channel_in, up_channel, kernel_size=7, bias=True),
                     nn.InstanceNorm2d(up_channel),
                     nn.ReLU(inplace=True))

        downsample0 = conv_down(up_channel)
        downsample1 = conv_down(downsample0.channel_output)

        resblock0 = ResBlock(downsample1.channel_output)
        resblock1 = ResBlock(downsample1.channel_output)
        resblock2 = ResBlock(downsample1.channel_output)

        down = nn.Sequential(downsample0, downsample1)

        res = nn.Sequential(resblock0, resblock1, resblock2)

        self.model = nn.Sequential(model, down, res)

    def forward(self, x):

        x = self.model(x)
        mu = self.shared_layer(x)
        noise = Variable(torch.randn(mu.size()).to(device))

        return mu, noise


class Decoder(nn.Module):
    def __init__(self, channel_in, channel_out, up_channel, shared_layer):
        super(Decoder, self).__init__()

        self.shared_layer = shared_layer

        resblock0 = ResBlock(channel_in)
        resblock1 = ResBlock(channel_in)
        resblock2 = ResBlock(channel_in)

        upsample0 = conv_up(channel_in)
        upsample1 = conv_up(upsample0.channel_output)

        output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(upsample1.channel_output, channel_out, kernel_size=7),
            nn.Tanh()
        )

        res = nn.Sequential(self.shared_layer, resblock0, resblock1, resblock2)

        up = nn.Sequential(upsample0, upsample1)

        output = nn.Sequential(output)

        self.model = nn.Sequential(res, up, output)

    def forward(self, x):

        return self.model(x)

class conv_down(nn.Module):

    def __init__(self, channel_input, channel_output = 16, kernel=3, stride=1, padding=0, Norm=True, Dropout=0.0):
        super(conv_down, self).__init__()

        self.channel_output = channel_input * 2

        steps = [nn.Conv2d(channel_input, self.channel_output, kernel_size=kernel, stride=stride, padding=padding, bias=False)]

        if Norm:
            steps.append(nn.InstanceNorm2d(self.channel_output))

        steps.append(nn.ReLU(inplace=True))

        if Dropout > 0:
            steps.append(nn.Dropout(Dropout))

        self.model = nn.Sequential(*steps)

    def forward(self, x):

        return self.model(x)

class conv_up(nn.Module):

    def __init__(self, channel_input, channel_output = None, kernel=3, stride=1, padding=0, Norm=True, Dropout=0):
        super(conv_up, self).__init__()


        self.channel_output = channel_input // 2

        steps = [nn.ConvTranspose2d(channel_input, self.channel_output, kernel_size=kernel, stride=stride, padding=padding, bias=False)]

        if Norm:
            steps.append(nn.InstanceNorm2d(self.channel_output))
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
    def forward(self, x):

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

class discriminator(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(discriminator, self).__init__()

        self.patch_discrim = nn.Sequential(
            nn.Conv2d(in_channel, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 1, 4, 1, 1)
        )
    def forward(self, x):

        x = self.patch_discrim(x)

        return x



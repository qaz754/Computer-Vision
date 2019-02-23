
import torch
import torch.nn as nn

from util import Flatten

class discriminator(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(discriminator, self).__init__()

        self.out_channel = out_channel

        self.model = nn.Sequential(
            nn.Conv2d(in_channel, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
        )
        self.src = nn.Sequential(
            nn.Conv2d(128, 1, 4, 1, 1)
        )
        self.cls = nn.Sequential(
            nn.Conv2d(128, out_channel, 5, 1, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return self.src(x), self.cls(x).view(-1, self.out_channel)


def if_tensor(tensor):

    if isinstance(tensor, torch.Tensor):
        return True

#TODO Resnet block
#TODO Convnet block

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

class ResNet(nn.Module):
    def __init__(self, channel_in, channel_out, up_channel):
        super(ResNet, self).__init__()

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
        resblock3 = ResBlock(downsample1.channel_output)
        resblock4 = ResBlock(downsample1.channel_output)
        resblock5 = ResBlock(downsample1.channel_output)

        upsample0 = conv_up(downsample1.channel_output)
        upsample1 = conv_up(upsample0.channel_output)


        output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(upsample1.channel_output, channel_out, kernel_size=7),
            nn.Tanh()
        )


        down = nn.Sequential(downsample0, downsample1)

        res = nn.Sequential(resblock0, resblock1, resblock2, resblock3, resblock4, resblock5)

        up = nn.Sequential(upsample0, upsample1)

        output = nn.Sequential(output)

        self.model = nn.Sequential(model, down, res, up, output)

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

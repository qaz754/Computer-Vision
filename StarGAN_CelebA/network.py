

#input is 224

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def if_tensor(tensor):

    if isinstance(tensor, torch.Tensor):
        return True

class ResBlock(nn.Module):

    def __init__(self, input_channel):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(input_channel, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channel, input_channel, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(input_channel, affine=True, track_running_stats=True)
        )

    def forward(self, x):

        x = x + self.block(x)

        return x

class ResNet(nn.Module):
    def __init__(self, opts):
        super(ResNet, self).__init__()

        in_ch = opts.channel_up

        steps = []

        steps += [
                 nn.Conv2d(opts.channel_in + opts.num_classes, in_ch, 7, 1, 3, bias=False),
                 nn.InstanceNorm2d(in_ch, affine=True, track_running_stats=True),
                 nn.ReLU(inplace=True)]

        for i in range(opts.n_conv_down):
            steps.append(conv_down(in_ch))
            in_ch *= 2

        for i in range(opts.n_resblock):
            steps.append(ResBlock(in_ch))

        for i in range(opts.n_conv_up):
            steps.append(conv_up(in_ch))
            in_ch = in_ch // 2

        steps += [nn.Conv2d(in_ch, opts.channel_out, 7, 1, 3, bias=False)]

        if opts.G_out_activation == 'tanh':
            final_activation = nn.Tanh()
        elif opts.G_out_activation == 'sigm':
            final_activation = nn.Sigmoid()

        steps += [final_activation]

        self.model = nn.Sequential(*steps)

    def forward(self, x, c):

        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        return self.model(x)

class conv_down(nn.Module):

    def __init__(self, channel_input, kernel=4, stride=2, padding=1, Norm=True, Dropout=0.0):
        super(conv_down, self).__init__()

        self.channel_output = channel_input * 2

        steps = [nn.Conv2d(channel_input, self.channel_output, kernel_size=kernel, stride=stride, padding=padding, bias=False)]

        if Norm:
            steps.append(nn.InstanceNorm2d(self.channel_output, affine=True, track_running_stats=True))

        steps.append(nn.ReLU(inplace=True))

        if Dropout > 0:
            steps.append(nn.Dropout(Dropout))

        self.model = nn.Sequential(*steps)

    def forward(self, x):

        return self.model(x)

class conv_up(nn.Module):

    def __init__(self, channel_input, kernel=4, stride=2, padding=1, Norm=True, Dropout=0):
        super(conv_up, self).__init__()


        self.channel_output = channel_input // 2

        steps = [nn.ConvTranspose2d(channel_input, self.channel_output, kernel_size=kernel, stride=stride, padding=padding, bias=False)]

        if Norm:
            steps.append(nn.InstanceNorm2d(self.channel_output, affine=True, track_running_stats=True))
        steps.append(nn.ReLU(inplace=True))

        if Dropout > 0:
            steps.append(nn.Dropout(Dropout))

        self.model = nn.Sequential(*steps)

    def forward(self, x, skip_input=None):

        if if_tensor(skip_input):
            x = torch.cat((x, skip_input), 1)

        return self.model(x)

class discriminator(nn.Module):

    def __init__(self, opts):
        super(discriminator, self).__init__()

        self.opts = opts
        steps = []
        in_channel = opts.D_input_channel
        channel_up = opts.D_channel_up
        for i in range(opts.n_discrim_down):
            steps += [nn.Conv2d(in_channel, channel_up, 4, 2, 1), nn.LeakyReLU(opts.lrelu_val)]
            in_channel = channel_up
            channel_up *= 2

        kernel_size = int(opts.image_shape / (2 ** opts.n_discrim_down))

        src = [nn.Conv2d(in_channel, 1, 3, 1, 1, bias=False)]
        cls = [nn.Conv2d(in_channel, opts.num_classes, kernel_size, bias=False)]

        self.model = nn.Sequential(*steps)
        self.src = nn.Sequential(*src)
        self.cls = nn.Sequential(*cls)

    def forward(self, x):

        x = self.model(x)

        return self.src(x), self.cls(x).view((-1, self.opts.num_classes))


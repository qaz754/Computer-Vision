import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class double_conv_down(nn.Module):

    def __init__(self, channel_input, channel_output):
        super(double_conv_down, self).__init__()

        self.skip_output = nn.Sequential(
            nn.Conv2d(channel_input, channel_output, 3, stride=1),  # batch_size * 16 * 10 * 10
            nn.ReLU(True),
            nn.Conv2d(channel_output, channel_output, 3, stride=1),  # batch_size * 16 * 10 * 10
            nn.ReLU(True)
        )

        self.down_output = nn.Sequential(
            nn.MaxPool2d(2, stride=2),  # batch_size * 16 * 5 * 5
        )

    def forward(self, x):

        skip_output = self.skip_output(x)

        down_output = self.down_output(skip_output)

        return down_output, skip_output

class double_conv_up(nn.Module):

    def __init__(self, channel_input, channel_output, final_channel=None):
        super(double_conv_up, self).__init__()

        if final_channel == None:
            final_channel = channel_output

        self.up_output = nn.Sequential(
            nn.ConvTranspose2d(channel_input, channel_output, 3, stride=1),  # batch_size * 8 * 15 * 15
            nn.ReLU(True),
            nn.ConvTranspose2d(channel_output, final_channel, 3, stride=1),  # batch_size * 8 * 15 * 15
        )


    def forward(self, up_input, skip_input):

        x = torch.cat((up_input, skip_input), 1)
        x = self.up_output(x)

        return x

class AutoEncoder_Unet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AutoEncoder_Unet, self).__init__()

        self.U_down1 = double_conv_down(channel_input=in_channel, channel_output=16)
        self.U_down2 = double_conv_down(channel_input=16, channel_output=32)

        self.U_up1 = double_conv_up(64, 32, 16)
        self.U_up2 = double_conv_up(32, 8, out_channel)

        self.double_up0 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, 2, stride=2),
            nn.ReLU(True),
        )

        self.double_up1 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.ReLU(True),
        )

        self.tanh = nn.Sequential(
            nn.Tanh()
        )

    def forward(self, x):

        input_down1, input_skip1 = self.U_down1(x)
        input_down2, input_skip2 = self.U_down2(input_down1)

        input_down2 = self.double_up1(input_down2)
        input_up = self.U_up1(input_down2, input_skip2)

        input_up = self.double_up0(input_up)
        input_up = self.U_up2(input_up, input_skip1)

        return self.tanh(input_up)





class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()

        # define the layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # batch_size * 16 * 10 * 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # batch_size * 16 * 5 * 5
            nn.Conv2d(16, 8, 3, stride=2, padding=2),  # batch_size * 8 * 3 * 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)  # batch_size * 8 * 2 * 2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # batch_size * 16 * 5 * 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # batch_size * 8 * 15 * 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # batch_size * 1 * 28 * 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)


        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def if_tensor(tensor):

    if isinstance(tensor, torch.Tensor):
        return True

class Decoder(nn.Module):
    def __init__(self, encoder, depth=21, decoder_activation=nn.ReLU()):
        super(Decoder, self).__init__()

        self.encoder = encoder
        self.depth = depth
        self.decoder_activation = decoder_activation
        self.model = self.create()


    def create(self):

        model = []

        for i in reversed(range(self.depth)):
            layer = self.encoder.features[i]
            if 'Conv' in str(layer):
                in_channel, out_channel = layer.out_channels,  layer.in_channels
                model.append(nn.ReflectionPad2d((1)))
                model.append(nn.Conv2d(in_channel, out_channel, (3, 3)))
                model.append(self.decoder_activation)
            elif 'MaxPool' in str(layer):
                #model.append(nn.Upsample(scale_factor=2))
                model.append(Interpolate(size=2, mode='nearest'))

        model.pop() #get rid of the last ReLU

        return nn.Sequential(*model)

    def forward(self, ADaIN_out, c_feature, alpha=1):

        x = (1 - alpha) * c_feature + alpha * ADaIN_out

        return self.model(x)

class Interpolate(nn.Module):
    #since nn.UpsamplingNearest2d is deprecated.
    #https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.size, mode=self.mode, align_corners=None)
        return x

class AdaIN():
    def __init__(self, eps= 1e-5):

        self.eps = eps


    def mean_std(self, x):

        shape = x.size()

        N, C = shape[:2]

        style_reshaped = x.view(x.size(0), x.size(1), -1)
        target_std = style_reshaped.var(2) + self.eps
        target_std = target_std.sqrt().view(N, C, 1, 1)
        target_mean = style_reshaped.mean(2).view(N, C, 1, 1)

        return target_mean, target_std

    def ADaIN(self, x):

        content = x[0]
        style = x[1]

        size = content.size()

        style_mean, style_std = self.mean_std(style)
        content_mean, content_std = self.mean_std(content)

        norm = (content - content_mean.expand(size)) / content_std.expand(size)

        return norm * style_mean.expand(size) + style_mean.expand(size)






"""
class AdaIN(nn.Module):
    def __init__(self, nOutput, disabled=False, eps= 1e-5):
        super(AdaIN, self).__init__()

        self.nOutput = nOutput
        self.disabled = disabled
        self.eps = eps

    def cal_mean_std(self, x, loss=True):

        N = x.size(0)

        Hs, Ws = x.size(2), x.size(3)

        if loss:
            style_reshaped = x.view(x.size(0), x.size(1), Hs * Ws)
            target_std = style_reshaped.var(2).sqrt().view(N, -1) + self.eps
        else:
            style_reshaped = x.view(x.size(0), self.nOutput, Hs * Ws)
            target_std = style_reshaped.var(2).sqrt().view(N, -1)

        target_mean = style_reshaped.mean(2).view(N, -1)

        return target_mean, target_std

    def forward(self, x):
        '''
        Takes in a tuple of content and style images and outputs a stylized content image
        :param x: A tuple of tensor {Content, Style}
        :return: Content
        '''

        content = x[0]
        style = x[1]

        if self.disabled:
            return content

        N = content.size(0)

        Hc, Wc = content.size(2), content.size(3)

        target_mean, target_std = self.cal_mean_std(style, False)

        '''create instancenorm with the right demensions and copy over the target std and mean '''
        norm = nn.BatchNorm1d(N* self.nOutput, self.eps).to(device)

        with torch.no_grad():
            norm.weight = torch.nn.Parameter(target_std)
            norm.bias = torch.nn.Parameter(target_mean)

        content_reshaped = content.view(1, N*self.nOutput, Hc*Wc)
        output = norm(content_reshaped).view(content.size())

        return output
"""
'''
EXAMPLE
qq = AdaIN(3, False)
f = qq.forward([b, a])
import util
fpred_image_list = []
directory = './img/'
fpred_image_list.append(util.save_images_to_directory(f, directory, 'x1_2_recon_image_%s.png' % 0))
fpred_image_list.append(util.save_images_to_directory(b, directory, 'x1_2_recon_image_%s.png' % 1))
'''


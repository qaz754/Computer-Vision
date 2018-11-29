
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self, kernal_size, output_size, drop_p = 0.5):
        super(Network, self).__init__()

        self.kernal_size = kernal_size
        self.output_size = output_size

        # 1 color channel
        # 10 output channel
        # 3 X 3 conv
        self.conv1 = nn.Conv2d(1, 10, kernal_size)
        #kernel_size = 2, stride = 2
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, kernal_size)
        self.fc1 = nn.Linear(20 * 5 * 5, 50)
        self.fc1_drop = nn.Dropout(p =drop_p)
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, data):

        x = self.pool(F.relu(self.conv1(data)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)

        return x

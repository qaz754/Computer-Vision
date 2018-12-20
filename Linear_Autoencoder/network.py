
import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self, input_size, hidden_1, hidden_2, hidden_3, bottle_neck):
        super(AutoEncoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(input_size, hidden_1),
            nn.ReLU(True),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(True),
            nn.Linear(hidden_2, hidden_3),
            nn.ReLU(True),
            nn.Linear(hidden_3, bottle_neck)
        )
        self.decode = nn.Sequential(
            nn.Linear(bottle_neck, hidden_3),
            nn.ReLU(True),
            nn.Linear(hidden_3, hidden_2),
            nn.ReLU(True),
            nn.Linear(hidden_2, hidden_1),
            nn.ReLU(True),
            nn.Linear(hidden_1, input_size),
            nn.Tanh()
        )

    def forward(self, x):

        x = self.encode(x)
        x = self.decode(x)

        #reshape the output to B * 1 * 28 * 28
        return x.view((-1, 1, 28, 28))


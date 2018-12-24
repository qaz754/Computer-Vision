
import torch
import torch.nn as nn
def sample_noise(batch_size, dim):
    """
    Given the inputs batch_size, and dim return a tensor of values between (L, U)
    :param batch_size (int): size of the batch
    :param dim (int): size of the vector of dimensions
    :return: tensor of a random values between (L, U)

    Used to generate images in GANs
    """

    #TODO take the L and U as inputs of the function
    L = -1 #lower bound
    U = 1 #upper bound

    noise = (L - U) * torch.rand((batch_size, dim)) + U

    return noise

class Flatten(nn.Module):
    """
    Given a tensor of Batch * Color * Height * Width, flatten it and make it 1D.
    Used for Linear GANs
    """
    def forward(self, x):

        B, C, H, W = x.size()

        return x.view(B, -1) #returns a vector that is B * (C * H * W)




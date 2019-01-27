from collections import deque, namedtuple
import numpy as np
import torch
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["image0", "image1", "image2", "image2_pred"])
        self.seed = random.seed(seed)

    def add(self, image0, image1, image2, image2_pred):
        """Add a new experience to memory."""

        e = self.experience(image0, image1, image2, image2_pred)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        experiences = random.sample(self.memory, k=len(self.memory))

        image0 = torch.cat([e.image0 for e in experiences if e is not None])
        image1 = torch.cat([e.image1 for e in experiences if e is not None])
        image2 = torch.cat([e.image2 for e in experiences if e is not None])
        image2_pred = torch.cat([e.image2_pred for e in experiences if e is not None])


        return image0, image1, image2, image2_pred

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
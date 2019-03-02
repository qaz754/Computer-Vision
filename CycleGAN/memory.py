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
        self.experience = namedtuple("Experience", field_names=["image"])
        self.seed = random.seed(seed)

    def add(self, image):
        """Add a new experience to memory."""
        e = self.experience(image)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        experiences = random.sample(self.memory, k=len(self.memory))
        image = torch.cat([e.image for e in experiences if e is not None])
        #image = [e.image for e in experiences if e is not None]

        return image

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
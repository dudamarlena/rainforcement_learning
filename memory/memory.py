""" Module with agent Memory """

from collections import deque
import random
import numpy as np
import paramethers


class Memory(object):
    def __init__(self, buffer_size, random_seed=paramethers.RANDOM_SEED):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, experience):
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        states = np.array([row[0] for row in batch])
        actions = np.array([row[1] for row in batch])
        rewards = np.array([row[2] for row in batch])
        done = np.array([row[3] for row in batch])
        next_states = np.array([row[4] for row in batch])
        return states, actions, rewards, done, next_states

    def clear(self):
        self.buffer.clear()
        self.count = 0


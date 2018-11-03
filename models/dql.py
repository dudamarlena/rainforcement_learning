""" Module with Deep Q Network model and Memory of experience """

from tensorflow.keras import (
    Sequential,
    optimizers,
)
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
)
from collections import deque
import numpy as np

from experience import Experience
import paramethers as param


class DeepQNetwork:
    """
    Class with Deep Q-Network model
     * Dense layer [32] -> Batch Normalization -> relu
     * Dense layer [64] -> Batch Normalization -> relu
     * Dense layer [64] -> Batch Normalization -> relu
     * Dense layer [6] -> sigmoid
     * Optimizer: Adam
     * Loss: Mean squared error
    """
    def __init__(self, env):
        self.env = env
        self.action_size = param.ACTION_NUMBER

        self.model = Sequential()
        self.model.add(Dense(32, activation='relu', input_shape=self.env.observation_space.shape, ))
        self.model.add(BatchNormalization())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(BatchNormalization())
        # self.model.add(Dense(64, activation='relu'))
        # self.model.add(BatchNormalization())
        self.model.add(Dense(self.action_size, activation='sigmoid'))
        self.optimizer = optimizers.Adam(lr=param.LEARNING_RATE)
        self.model.compile(loss='mse', optimizer=self.optimizer)


class Memory:
    """
    Memory of agent which store his all experiences in each state
    """
    def __init__(self):
        self.buffer = deque()

    def add(self, experience: Experience):
        """
        Add a new experience to memory
        :param experience: agent experience
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int = 32):
        """
        Get list of sample experiences
        :param batch_size: size of batch which is length of list
        :return: list of examples
        """
        index = np.random.choice(
            np.arange(len(self.buffer)),
            size=batch_size,
        )
        return [self.buffer[i] for i in index]

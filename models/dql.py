""" Module with Deep Q Network model and Memory of experience """

from tensorflow.keras import (
    Sequential,
    optimizers,
)
from tensorflow.keras.layers import Dense

import paramethers as param


class DeepQNetwork:
    """
    Class with Deep Q-Network model
     * Dense layer [64] -> relu
     * Dense layer [64] -> relu
     * Dense layer [64] -> relu
     * Dense layer [6] -> tanh
     * Optimizer: Adam
     * Loss: Mean squared error
    """
    def __init__(self, env):
        self.env = env
        self.action_size = env.action_space.shape[0]

        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_shape=self.env.observation_space.shape, ))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(self.action_size, activation='tanh'))
        self.optimizer = optimizers.Adam(lr=param.ACTOR_LR)
        self.model.compile(loss='mse', optimizer=self.optimizer)

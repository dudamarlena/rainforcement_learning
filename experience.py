""" Module with Experience of agent getting in each state """

from typing import NamedTuple
import numpy as np


class Experience(NamedTuple):
    """ Class with all experience fields """
    state: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    next_state: np.ndarray

    @classmethod
    def create_experience(cls, state, action, reward, done, next_state) -> 'Experience':
        """
        Function to fill Experience fields
        """
        return cls(
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
        )

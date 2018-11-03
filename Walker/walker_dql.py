""" Module to train 'Walker' by DQL algorithm """
#DDPG
from typing import Dict
from random import randrange
import gym
import numpy as np
import pybullet_envs

import paramethers as param
from experience import Experience
from models.dql import (
    DeepQNetwork,
    Memory,
)
from Walker import config


def main():
    """
    Main function to learn 'Walker' walking
     * step1: pre-training
     * step2: training
    """
    env = gym.make(config.WALKER)
    env.render(mode=config.MODEL)
    dqn = DeepQNetwork(env=env)
    memory = Memory()

    # pre_training(env, memory)

    # env = gym.wrappers.Monitor(env, config.MONITOR_PATH)
    training(env, dqn, memory)


# def pre_training(env: gym.Env, memory: Memory):
#     """
#     Pre-training the Walker by used random actions
#     :param env: Walker environment
#     :param memory: memory of Walker with all saved experiences
#     """
#     env.reset()
#     random_action = env.action_space.sample()
#     state, reward, done, info = env.step(random_action)
#
#     for i in range(param.PRETRAIN_EPISODES):
#
#         random_action = env.action_space.sample()
#         next_state, reward, done, info = env.step(random_action)
#         experience = Experience.get_experience(state, random_action, reward, done, next_state)
#         if done:
#             state = _failed_simulation(env, experience, memory)
#         else:
#             state = _make_next_move(env, experience, memory)


def training(
        env: gym.Env,
        dqn: DeepQNetwork,
        memory: Memory,
):
    """
    Train Walker by using Deep Q Learning network and updating Q-values
    :param env: Walker environment
    :param dqn: Deep Q-network model
    :param memory: memory of Walker with all saved experiences
    """
    step = 0
    possible_actions = _possible_actions(env)
    for episode in range(param.TRAIN_EPISODES):
        env.reset()
        random_action = possible_actions[randrange(param.ACTION_NUMBER)]
        state, reward, done, _ = env.step(random_action)
        state = np.reshape(state, (1,) + env.observation_space.shape)
        total_reward = 0

        done = False
        while not done:
            env.render()
            action_num_, explore_prob = _choice_action(env, dqn, step, state)

            action = possible_actions[action_num_]
            next_state, reward, done, info = env.step(action)
            experience = Experience.create_experience(
                state=state,
                action_number=action_num_,
                action=action,
                reward=reward,
                done=done,
                next_state=next_state
            )
            total_reward += reward

            if done:
                if episode % (param.VIDEOREC_RATE / 10) == 0:
                    print(f'Episode num: {episode}',
                          f'Total reward: {total_reward}',
                          f'Exploration probabilistic: {explore_prob}')

                next_state = np.zeros(experience.state.shape)
                _save_experience(experience, memory, next_state)

            else:
                next_state = np.reshape(next_state, (1,) + env.observation_space.shape)
                _save_experience(experience, memory, next_state)
                state = next_state

            update_model(dqn, env, memory)
        step += 1


def update_model(dqn: DeepQNetwork, env: gym.Env, memory: Memory):
    """
    Update Q-table and fit model
    :param dqn: Deep Q-network model
    :param env: Walker Environment
    :param memory: Memory of agent
    """
    inputs = np.zeros((param.BATCH_SIZE,) + env.observation_space.shape)
    targets = np.zeros((param.BATCH_SIZE, param.ACTION_NUMBER))

    for i, (state_, action_num_, _, reward_, _, next_state_) in enumerate(memory.sample()):
        inputs[i:i + 1] = state_
        targets[i] = dqn.model.predict(state_)
        q_state = dqn.model.predict(next_state_)[0]

        if (next_state_ == np.zeros(state_.shape)).all():
            targets[i][action_num_] = reward_
        else:
            targets[i][action_num_] = reward_ + param.GAMMA * np.amax(q_state)
    dqn.model.fit(inputs, targets, epochs=1, verbose=0)


def _save_experience(experience: Experience, memory: Memory, next_state: np.ndarray):
    """
    Save a new experience by replaced next state of agent
    :param next_state: nest state of agent
    :param experience: Experience of made action
    :param memory: memory of Walker with all saved experiences
    :return: next state which agent will make
    """
    experience = experience._replace(next_state=next_state)
    memory.add(experience)


def _choice_action(
        env: gym.Env,
        dqn: DeepQNetwork,
        step: int,
        state: np.ndarray,
) -> (np.ndarray, float):
    """
    Choice the action which agent will make
    :param env: Walker environment
    :param dqn: Deep Q-Network model
    :param step: number of iteration
    :param state: state in which agent is
    :return: if exploration probability is bigger than random value, return random action
     else choose action using DQL model
    """
    explore_prob = param.MIN_EXPLORE + (param.MAX_EXPLORE - param.MIN_EXPLORE) * \
                                       np.exp(-param.DECAY_RATE * step)
    if explore_prob > np.random.rand():
        action_num = randrange(param.ACTION_NUMBER)
    else:
        q_state = dqn.model.predict(state)[0]
        action_num = np.argmax(q_state)
    return action_num, explore_prob


def _possible_actions(
        env: gym.Env,
) -> Dict[int, np.ndarray]:
    """
    Generate list with possible action which agent could make
    :param env: Walker environment
    :return: list of actions
    """
    return {i: env.action_space.sample() for i in range(param.ACTION_NUMBER)}


if __name__ == "__main__":
    main()

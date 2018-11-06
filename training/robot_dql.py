""" Module to train robot walking by DQL algorithm """

import gym
import numpy as np
import pybullet_envs

import paramethers as param
from experience import Experience
from models.dql import DeepQNetwork
from models.memory import Memory
from training import config


def main(robot_name: str, env_monitor: bool = True):
    """
    Main function to learn robot walking using DQL algorithm
    """
    env = gym.make(robot_name)
    env.render(mode=config.MODEL)
    dqn = DeepQNetwork(env=env)
    memory = Memory(param.MEMORY_SIZE)
    if env_monitor:
        env = gym.wrappers.Monitor(
            env=env,
            directory=config.MONITOR_PATH,
            force=True,
            video_callable=lambda episode_id: episode_id % 1 == 0
        )
    print(f'Start training agent: {config.WALKER}')
    training(env, dqn, memory)


def training(
        env: gym.Env,
        dqn: DeepQNetwork,
        memory: Memory,
):
    """
    Train robot by using Deep Q Learning network and updating Q-values
    :param env: robot environment
    :param dqn: Deep Q-network model
    :param memory: memory of robot with all saved experiences
    """
    all_actions = 0
    step = 0
    for episode in range(param.TRAIN_EPISODES):
        env.reset()
        random_action = env.action_space.sample()
        state, reward, done, _ = env.step(random_action)
        state = np.reshape(state, (1,) + env.observation_space.shape)
        total_reward = 0

        actions_num = 0
        done = False
        while not done:
            env.render()
            action, explore_prob = _choice_action(env, dqn, step, state)

            next_state, reward, done, info = env.step(action)
            experience = Experience.create_experience(
                state=state,
                action=action,
                reward=reward,
                done=done,
                next_state=next_state
            )
            total_reward += reward
            actions_num += 1
            if done:
                all_actions += 0
                print(f'|Reward: {round(total_reward)} '
                      f'| Episode: {episode} '
                      f'| Qmax: {explore_prob}'
                      f'| Action number: {actions_num}'
                      f'| All actions {all_actions}')

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
    :param env: robot Environment
    :param memory: Memory of agent
    """
    inputs = np.zeros((param.BATCH_SIZE,) + env.observation_space.shape)
    targets = np.zeros((param.BATCH_SIZE, env.action_space.shape[0]))

    for i, (state_, action, reward_, _, next_state_) in enumerate(memory.sample()):
        inputs[i:i + 1] = state_
        targets[i] = dqn.model.predict(state_)
        q_state = dqn.model.predict(next_state_)[0]

        if (next_state_ == np.zeros(state_.shape)).all():
            targets[i][action] = reward_
        else:
            targets[i][action] = reward_ + param.GAMMA * np.amax(q_state)
    dqn.model.fit(inputs, targets, epochs=1, verbose=0)


def _save_experience(experience: Experience, memory: Memory, next_state: np.ndarray):
    """
    Save a new experience by replaced next state of agent
    :param next_state: nest state of agent
    :param experience: Experience of made action
    :param memory: memory of robot with all saved experiences
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
    If exploration probability is bigger than random value, return random action
     else choose action using DQL model
    :param env: robot environment
    :param dqn: Deep Q-Network model
    :param step: number of iteration
    :param state: state in which agent is
    :return: action and probability of exploration
    """
    explore_prob = param.MIN_EXPLORE + (param.MAX_EXPLORE - param.MIN_EXPLORE) * \
                   np.exp(-param.DECAY_RATE * step)
    if explore_prob > np.random.rand():
        action = env.action_space.sample()
    else:
        q_value = dqn.model.predict(state)[0]
        action = np.argmax(q_value)
    return action, explore_prob

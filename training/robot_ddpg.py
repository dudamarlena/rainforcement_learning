""" Module to train robot walking by DDPG algorithm """

import gym
import numpy as np
import pybullet_envs
import tensorflow as tf

from models.ddpg import Models
from models.memory import Memory
from experience import Experience
from training import config
import paramethers as param


def main(robot_name: str, env_monitor: bool = True):
    """
    Main function to create environment and teach the agent to walk
    :param robot_name: name of Robot Environment from PyBullet
    :param env_monitor: monitor the agent actions, default value: True
    """
    with tf.Session() as sess:
        env = gym.make(robot_name)
        _set_seed(env)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        models = Models.get_models(sess, state_dim, action_dim, action_bound)

        if env_monitor:
            env = gym.wrappers.Monitor(
                env=env,
                directory=config.MONITOR_PATH,
                force=True,
                video_callable=lambda episode_id: episode_id % 1 == 0
            )
        print(f'Start training agent: {config.WALKER}')
        train(sess, env, models)


def train(
        sess: tf.Session,
        env: gym.Env,
        models: Models,
        render: bool = False,
):
    """
    Train Walker by using DDPG algorithm
    :param sess: tenforflow session
    :param env: Environment of Walker
    :param models: tuple of used models
    :param render: render the move, default: true
    """
    if render:
        env.render(mode=config.MODEL)

    sess.run(tf.global_variables_initializer())

    models.actor.update_target_network()
    models.critic.update_target_network()

    memory = Memory(param.MEMORY_SIZE)
    memory.clear()
    all_actions = 0

    for episode in range(param.TRAIN_EPISODES):
        state = env.reset()
        episode_reward = 0
        episode_q_max = 0
        actions_num = 0
        done = False
        while not done:
            env.render()
            action = _make_action(models, state)[0]
            next_state, reward, done, _ = env.step(action)
            experience = Experience.create_experience(
                state=state,
                action=np.reshape(action, (models.actor.a_dim,)),
                reward=reward,
                done=done,
                next_state=next_state
            )
            memory.add(experience)

            if memory.size() > param.BATCH_SIZE:
                episode_q_max = update_targets(memory, models, episode_q_max)
            state = next_state
            episode_reward += reward

            if done:
                all_actions += actions_num
                print(f'|Reward: {round(episode_reward)} '
                      f'| Episode: {episode} '
                      f'| Qmax: {round(episode_q_max/actions_num, 2)}'
                      f'| Action number: {actions_num}'
                      f'| All actions {all_actions}')
                break
            actions_num += 1


def update_targets(memory: Memory, models: Models, episode_q_max: float) -> float:
    """
    Use Action and Critic Networks and update targets of both models
    :param memory: Memory of agent
    :param models: Used models
    :param episode_q_max: Max value of Q in episode
    """
    actor = models.actor
    critic = models.critic

    states_b, actions_s, rewards_b, done_b, next_states_b = memory.sample_batch(
        param.BATCH_SIZE
    )

    target = critic.predict_target(
        next_states_b, actor.predict_target(next_states_b)
    )

    y_i = []
    for index in range(param.BATCH_SIZE):
        if done_b[index]:
            y_i.append(rewards_b[index])
        else:
            y_i.append(rewards_b[index] + param.GAMMA * target[index])

    predicted_q_value, _ = critic.train(
        states_b, actions_s, np.reshape(y_i, (param.BATCH_SIZE, 1))
    )
    episode_q_max += np.amax(predicted_q_value)
    a_outs = actor.predict(states_b)
    grads = critic.action_gradients(states_b, a_outs)
    actor.train(states_b, grads[0])

    actor.update_target_network()
    critic.update_target_network()
    return episode_q_max


def _set_seed(env, seed=param.RANDOM_SEED):
    """ Set seed of generator of random numbers"""
    np.random.seed(seed)
    tf.set_random_seed(seed)
    env.seed(seed)


def _make_action(models: Models, state: np.ndarray):
    """ Predict action which agent make. Add Actor Noise to predicted action """
    return models.actor.predict(
        np.reshape(state, (1, models.actor.s_dim)
                   )) + models.actor_noise()

""" Module with Actor and Critic Networks """
import numpy as np
import tensorflow as tf
import tflearn
from typing import NamedTuple
import paramethers as param

UNIFORM_VALUE = 0.03


class Models(NamedTuple):
    """ Class with all needed models: Actor, Critic and ActorNoise """
    actor: ActorNetwork
    critic: CriticNetwork
    actor_noise: OrnsteinUhlenbeckActionNoise

    @classmethod
    def get_models(cls, sess, state_dim, action_dim, action_bound) -> 'Models':
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound)
        critic = CriticNetwork(sess, state_dim, action_dim, actor.get_num_trainable_vars())
        actor_noise = OrnsteinUhlenbeckActionNoise(np.zeros(action_dim))
        return cls(
            actor=actor,
            critic=critic,
            actor_noice=actor_noise
        )


class ActorNetwork:
    """

    """

    def __init__(self, sess, state_dim, action_dim, action_bound):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = param.ACTOR_LR
        self.tau = param.TAU
        self.batch_size = param.BATCH_SIZE

        self.inputs, self.out, self.scaled_out = self.create_actor_network()
        self.network_params = tf.trainable_variables()
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]
        self.update_target_network_params = self._update_params()
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
        self.actor_gradients = self._get_actor_gradient()

        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
            zip(self.actor_gradients, self.network_params)
        )
        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 64)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 64)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        w_init = tflearn.initializations.uniform(minval=-UNIFORM_VALUE, maxval=UNIFORM_VALUE)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init
        )
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(
            self.optimize,
            feed_dict={
                self.inputs: inputs,
                self.action_gradient: a_gradient
            }
        )

    def predict(self, inputs):
        return self.sess.run(
            self.scaled_out,
            feed_dict={
                self.inputs: inputs
            }
        )

    def predict_target(self, inputs):
        return self.sess.run(
            self.target_scaled_out,
            feed_dict={
                self.target_inputs: inputs
            }
        )

    def update_target_network(self):
        self.sess.run(
            self.update_target_network_params
        )

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def _update_params(self):
        return [
            self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) +
                tf.multiply(self.target_network_params[i], 1. - self.tau)
            )
            for i in range(len(self.target_network_params))]

    def _get_actor_gradient(self):
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out,
            self.network_params,
            -self.action_gradient
        )
        return list(map(
            lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients
        ))


class CriticNetwork:
    """

    """

    def __init__(self, sess, state_dim, action_dim, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = param.CRITIC_LR
        self.tau = param.TAU

        self.inputs, self.action, self.out = self.create_critic_network()
        self.network_params = tf.trainable_variables()[num_actor_vars:]
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]
        self.update_target_network_params = self._update_params()

        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 64)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        t1 = tflearn.fully_connected(net, 64)
        t2 = tflearn.fully_connected(action, 64)

        net = tflearn.activation(
            incoming=tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b,
            activation='relu'
        )
        w_init = tflearn.initializations.uniform(minval=-UNIFORM_VALUE, maxval=UNIFORM_VALUE)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run(
            [self.out, self.optimize],
            feed_dict={
                self.inputs: inputs,
                self.action: action,
                self.predicted_q_value: predicted_q_value
            }
        )

    def predict(self, inputs, action):
        return self.sess.run(
            self.out, feed_dict={
                self.inputs: inputs,
                self.action: action
            }
        )

    def predict_target(self, inputs, action):
        return self.sess.run(
            self.target_out,
            feed_dict={
                self.target_inputs: inputs,
                self.target_action: action
            }
        )

    def action_gradients(self, inputs, actions):
        return self.sess.run(
            self.action_grads,
            feed_dict={
                self.inputs: inputs,
                self.action: actions
            }
        )

    def update_target_network(self):
        self.sess.run(
            self.update_target_network_params
        )

    def _update_params(self):
        return [
            self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) +
                tf.multiply(self.target_network_params[i], 1. - self.tau)
            )
            for i in range(len(self.target_network_params))
            ]


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2 * np.ones(6), theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

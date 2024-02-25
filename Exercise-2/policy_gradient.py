import logging

import gym
from datetime import datetime
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow as tf_v2
import collections
from keras.layers import Input, Dense, Dropout
from keras import Sequential
from keras.initializers.initializers import HeUniform
from csv_logger import CsvLogger

# from logger import logger.
# optimized for Tf2
tf.disable_v2_behavior()
print("tf_ver:{}".format(tf.__version__))

env = gym.make('CartPole-v1')
np.random.seed(1)


class ValueNetwork:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.v = self.init_v()

    def init_v(self):
        v = Sequential([
            Input(shape=(4,)),
            # Dense(units=32, activation='relu', kernel_initializer=HeUniform()),
            Dense(units=32, activation='relu', kernel_initializer=HeUniform(seed=42)),
            Dense(units=32, activation='relu', kernel_initializer=HeUniform(seed=42)),
            # Dense(units=16, activation='relu', kernel_initializer=HeUniform()),
            Dense(units=1, activation='linear')
        ])
        v.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mse")
        return v


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [12], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [12, self.action_size], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


def run(baseline_mode: bool):
    # Define hyperparameters
    state_size = 4
    action_size = env.action_space.n

    max_episodes = 5000
    max_steps = 1000
    discount_factor = 0.99
    learning_rate = 0.0008

    render = True

    # Initialize the policy network
    tf.reset_default_graph()
    policy = PolicyNetwork(state_size, action_size, learning_rate)
    valueNetwork = ValueNetwork(learning_rate)

    # init tensorboard
    log_file = "logs/ex2/" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
    # logger = Logger(log_dir + "-bpg-log.csv")
    csvlogger = CsvLogger(log_file, delimiter=',', level=logging.INFO,
                          header=['timestamp','step', 'RPE', 'AVG100', "LOSS"])
    # Start training the agent with REINFORCE algorithm
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        solved = False
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0

        for episode in range(max_episodes):
            baseline = list()
            state, _ = env.reset()
            state = state.reshape([1, state_size])
            episode_transitions = []

            for step in range(max_steps):
                actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                next_state, reward, done, info, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])

                # if render:
                #     # env.render()

                action_one_hot = np.zeros(action_size)
                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1
                episode_transitions.append(
                    Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
                episode_rewards[episode] += reward

                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                    #     tf.summary.scalar("Average Reward (Last 100 Episodes)", average_rewards)
                    # tf.summary.scalar("Reward Per Episode", episode_rewards[episode])
                    print(
                        "Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                     round(average_rewards, 2)))
                    if average_rewards > 475:
                        print(' Solved at episode: ' + str(episode))
                        solved = True
                    break
                state = next_state

            if solved:
                break

            transitions = []
            discounted_returns = []
            # Compute Rt for each time-step t and update the network's weights
            for t, transition in enumerate(episode_transitions):
                total_discounted_return = sum(
                    discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:]))  # Rt
                # something here
                if baseline_mode:

                    # baseline.append(total_discounted_return)
                    advantage = total_discounted_return - valueNetwork.v.predict(transition.state)

                    transitions.append(transition.state)
                    discounted_returns.append(total_discounted_return)
                    feed_dict = {policy.state: transition.state, policy.R_t: advantage,
                                 policy.action: transition.action}
                    _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
                else:
                    feed_dict = {policy.state: transition.state, policy.R_t: total_discounted_return,
                                 policy.action: transition.action}
                    _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)
            if baseline_mode:
                valueNetwork.v.train_on_batch(
                    np.vstack(transitions),
                    np.vstack(discounted_returns)
                )
            csvlogger.info(
                [
                    episode,
                    episode_rewards[episode],
                    average_rewards,
                    loss
                ]
            )


if __name__ == '__main__':
    run(baseline_mode=True)

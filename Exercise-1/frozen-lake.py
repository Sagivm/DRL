import gymnasium as gym
import numpy as np
import random
import copy
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import subprocess
import time

seed = 42
random.seed(seed)

# TensorBoard log directory
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# TensorBoard writer init.
file_writer = tf.summary.create_file_writer(log_dir)


# Start TensorBoard before running this script, via pycharm's terminal: tensorboard --logdir logs/fit
# Open TensorBoard -  http://localhost:6006/
# subprocess.Popen(["tensorboard", "--logdir", log_dir])
# time.sleep(1)

# Normalizes Q Tables and Plots TensorBoard Color-maps &
# Plots TensorBoard Scalars for Average Reward and Steps to Goal per 100 episodes
def plot_q_tables_and_log_rewards_and_steps_to_tensorboard(q_table, rewards, steps):
    def plot_q_table(q_table, which):
        actions = ["Up", "Right", "Down", "Left"]
        states = np.arange(0, 16)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(q_table.T, cmap='RdYlGn')

        ax.set_yticks(np.arange(len(actions)))
        ax.set_yticklabels(actions)
        plt.setp(ax.get_yticklabels(), ha="right", rotation_mode="anchor")

        # Display Q-Values
        for i in range(len(actions)):
            for j in range(len(states)):
                ax.text(j, i, np.round(q_table[j, i], 2), ha="center", va="center", color="b")

        ax.set_title(f"Q-value table - {which}")
        fig.tight_layout()

        cbar = fig.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label('Q-Value')

        plt.show()

    plot_q_table(q_table[499], 500)
    plot_q_table(q_table[1999], 2000)
    plot_q_table(q_table[-1], which="Final")

    with file_writer.as_default():
        for episode, (reward, step) in enumerate(zip(rewards, steps)):
            if episode % 100 == 0 and episode >= 100:
                avg_reward = np.mean(rewards[episode - 100:episode])
                tf.summary.scalar("Average Reward (Last 100 Episodes)", avg_reward, step=episode)
                avg_steps = np.mean(steps[episode - 100:episode])
                tf.summary.scalar("Average Steps (Last 100 Episodes)", avg_steps, step=episode)


env = gym.make('FrozenLake-v1', desc=None, map_name="4x4")


def initialize_q_table(state_space, action_space):
    """
        Initialize q table with zeros
        :param state_space:
        :type state_space:
        :param action_space:
        :type action_space:
        :return:
        :rtype:
        """

    q_table = np.zeros((state_space, action_space))
    return q_table


def epsilon_greedy_policy(q_table, state, epsilon: float):
    rand = random.uniform(0, 1)
    return np.argmax(q_table[state]) if rand > epsilon else env.action_space.sample()


def greedy_policy(q_table, state):
    return np.argmax(q_table[state])


def evaluate_in_chunk(steps, chunk_size):
    # Calculate the number of chunks
    num_chunks = len(steps) // chunk_size

    # Initialize a list to store the averages
    averages = []

    # Iterate through the chunks and calculate the average for each
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        chunk = steps[start_index:end_index]

        # Calculate the average for the current chunk
        average = sum(chunk) / len(chunk)

        # Append the average to the list
        averages.append(average)
    return averages


def train(q_table, n_training_episodes: int, epsilon: float, decay_rate: float, alpha: float,
          gamma, max_steps):
    q_tables = []

    for episode in range(n_training_episodes):
        state, prob = env.reset(seed=seed)
        # repeat
        done = False
        k = 0

        epsilon *= decay_rate

        for k in range(max_steps):
            # Sample action a
            action = epsilon_greedy_policy(q_table, state, epsilon)
            # Get next state s'
            new_state, reward, done, info, _ = env.step(action)

            if done:

                target = reward

                new_state, prob = env.reset(seed=seed)
                # break;
            else:
                target = reward + gamma * np.max(q_table[new_state, :])

            q_table[state][action] = (1 - alpha) * q_table[state][action] + alpha * target

            state = new_state
            if done:
                break
        if done and reward != 1:
            k = max_steps

        q_tables.append((reward, k, copy.deepcopy(q_table)))
    return q_tables


def main():
    # Training parameters
    n_training_episodes = 5000
    learning_rate = 0.1

    # Evaluation parameters
    n_eval_episodes = 100

    # Environment parameters
    max_steps = 100
    # Exploration parameters
    epsilon = 0.9
    decay_rate = 0.999
    gamma = 0.95
    eval_seed = []

    # env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

    print(env)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = initialize_q_table(n_states, n_actions)
    train_r = train(q_table, n_training_episodes, epsilon, decay_rate, learning_rate, gamma,
                    max_steps)
    rewards = list((map(lambda q: q[0], train_r)))
    steps = list((map(lambda q: q[1], train_r)))
    q_tables = list((map(lambda q: q[2], train_r)))
    # result = list(map(lambda q_table: evaluate_agent(100, q_table), q_tables))

    plot_q_tables_and_log_rewards_and_steps_to_tensorboard(q_tables, rewards, steps)


if __name__ == "__main__":
    main()

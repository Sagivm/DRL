import gymnasium as gym
import numpy as np
import random
import copy

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
    """
    Get next state from epsilon greedy policy
    :param q_table:
    :type q_table:
    :param state:
    :type state:
    :param epsilon:
    :type epsilon:
    :return:
    :rtype:
    """
    rand = random.uniform(0, 1)
    return np.argmax(q_table[state]) if rand > epsilon else env.action_space.sample()


def greedy_policy(q_table, state):
    """
    Get next state from greedy policy
    :param q_table:
    :type q_table:
    :param state:
    :type state:
    :return:
    :rtype:
    """
    return np.argmax(q_table[state])


def evaluate_avg_in_chunk(lst: list, chunk_size: int) -> list[float]:
    """
    Evaluate list in chunks according to chunk size
    :param lst:
    :type lst:
    :param chunk_size:
    :type chunk_size:
    :return:
    :rtype:
    """
    # Calculate the number of chunks
    num_chunks = len(lst) // chunk_size

    # Initialize a list to store the averages
    averages = []

    # Iterate through the chunks and calculate the average for each
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        chunk = lst[start_index:end_index]

        # Calculate the average for the current chunk
        average = sum(chunk) / len(chunk)

        # Append the average to the list
        averages.append(average)
    return averages


def train(q_table, n_training_episodes: int, max_epsilon: float, min_epsilon: float, decay_rate: float, alpha: float,
          gamma, max_steps):
    q_tables = []
    for episode in range(n_training_episodes):
        state, prob = env.reset()
        # repeat
        done = False
        k = 0

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        for k in range(max_steps):
            # Sample action a
            action = epsilon_greedy_policy(q_table, state, epsilon)
            # Get next state s'
            new_state, reward, done, info, _ = env.step(action)

            if done:

                target = reward

                new_state, prob = env.reset()
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
    max_epsilon = 0.9
    min_epsilon = 0.05
    decay_rate = 0.001
    gamma = 0.95

    print(env)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = initialize_q_table(n_states, n_actions)
    train_r = train(q_table, n_training_episodes, max_epsilon, min_epsilon, decay_rate, learning_rate, gamma,
                    max_steps)
    rewards = list((map(lambda q: q[0], train_r)))
    steps = list((map(lambda q: q[1], train_r)))
    q_tables = list((map(lambda q: q[2], train_r)))

    np.savetxt("500.csv", q_tables[499], delimiter=',')
    np.savetxt("2000.csv", q_tables[1999], delimiter=',')
    np.savetxt("final.csv", q_tables[-1], delimiter=',')
    np.savetxt('rewards.csv', evaluate_avg_in_chunk(lst=rewards,chunk_size=50), delimiter=",")
    np.savetxt('steps.csv', evaluate_avg_in_chunk(lst=steps, chunk_size=100), delimiter=",")


if __name__ == "__main__":
    main()

import gymnasium
import logging
import random
from datetime import datetime
import time
import collections
import numpy as np
from policy_gradients import PolicyNetwork
import tensorflow as tf
from tensorflow import keras


tf.keras.utils.disable_interactive_logging()  # Hide per-step log
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Hide TF warnings

# Optimize GPU's involvement
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

# TensorBoard log directory
log_dir = "logs/drl_ex2/" + datetime.now().strftime("%Y%m%d-%H%M%S")

# Start TensorBoard before running this script, via pycharm's terminal:
# tensorboard --logdir logs/drl_ex2
# or
# python -m tensorboard.main --logdir=drl_ex2
# Open TensorBoard -  http://localhost:6006/


seed = 42
random.seed(seed)
step_counter = 0


class CriticNetwork:
    """
    Critic Network class for approximating the state-value function.
    """

    def __init__(self, state_size, learning_rate):
        """
        Initializes the Critic Network.

        Args:
            state_size (int): Dimensionality of the state space.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.state_size = state_size
        self.learning_rate = learning_rate
        
        self.value_estimator = keras.Sequential([
            keras.layers.InputLayer(input_shape=(self.state_size,)),
            keras.layers.Dense(units=32, activation='swish', kernel_initializer=tf.keras.initializers.HeUniform()),
            keras.layers.Dense(units=32, activation='swish', kernel_initializer=tf.keras.initializers.HeUniform()),
            keras.layers.Dense(units=1, activation='linear')])
           
        self.value_estimator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mse")


def run():
    """
    Trains the agent using the REINFORCE algorithm.
    """
    tf.compat.v1.disable_eager_execution()
    env = gymnasium.make("CartPole-v1")

    # Define hyperparameters
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    max_episodes = 5000
    max_steps = 501
    discount_factor = 0.99
    
    render = False
    
    actor_lr = 0.001  # Actor Network's Learning Rate
    critic_lr = 0.001  # Critic Network's Learning Rate

    # Initialize Actor (Policy) and Critic Networks
    tf.compat.v1.reset_default_graph()
    actor_network = PolicyNetwork(state_size, action_size, actor_lr)
    critic_network = CriticNetwork(state_size, critic_lr)

    # Start training
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        solved = False
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0

        # TensorBoard writer init.
        summary_writer = tf.compat.v1.summary.FileWriter(log_dir)


        for episode in range(max_episodes):
            state, _ = env.reset(seed=seed)
            state = np.reshape(state, [1, state_size])
            episode_transitions = []

            for step in range(max_steps):
                global step_counter
                step_counter += 1
                actions_distribution = sess.run(actor_network.actions_distribution, {actor_network.state: state})
                action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                next_state, reward, done, _, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])

                if render:
                    env.render()

                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1
                episode_transitions.append(
                    Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
                episode_rewards[episode] += reward

                curr_val, next_val = critic_network.value_estimator.predict(state), critic_network.value_estimator.predict(next_state)
                td_error = reward + (1 - done) * discount_factor * next_val - curr_val

                critic_loss_per_step = critic_network.value_estimator.fit(state, np.atleast_2d(td_error + curr_val), verbose=0).history['loss'][0]

                input_data = {actor_network.state: state, actor_network.R_t: td_error, actor_network.action: action_one_hot}
                _, actor_loss_per_step = sess.run([actor_network.optimizer, actor_network.loss], input_data)

                summary_writer.add_summary(tf.compat.v1.Summary(
                    value=[tf.compat.v1.Summary.Value(tag="Actor Loss Per Step", simple_value=actor_loss_per_step)]),
                    global_step=step_counter)
                summary_writer.add_summary(tf.compat.v1.Summary(
                    value=[tf.compat.v1.Summary.Value(tag="Critic Loss Per Step", simple_value=critic_loss_per_step)]),
                    global_step=step_counter)

                if done:
                    break
                state = next_state

            summary_writer.add_summary(tf.compat.v1.Summary(
                value=[tf.compat.v1.Summary.Value(tag="Reward Per Episode",
                                                  simple_value=float(episode_rewards[episode]))]),
                global_step=episode)
            if episode > 98:
                average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                summary_writer.add_summary(tf.compat.v1.Summary(
                    value=[tf.compat.v1.Summary.Value(tag="Average Reward (Last 100 Episodes)",
                                                      simple_value=average_rewards)]), global_step=episode)
            print("Episode {} Reward: {} Average over 100 episodes: {}.\nActor Loss {} Critic Loss {}".format(episode, round(episode_rewards[episode], 2), round(average_rewards, 2), actor_loss_per_step, critic_loss_per_step))
            if average_rewards > 475:
                print(' Solved at episode: ' + str(episode))
                solved = True
                summary_writer.add_summary(tf.compat.v1.Summary(
                    value=[tf.compat.v1.Summary.Value(tag="Average Reward (Last 100 Episodes)",
                                                      simple_value=average_rewards)]), global_step=episode)

            if solved:
                break

        summary_writer.close()


if __name__ == '__main__':
    run()

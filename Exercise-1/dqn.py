import copy
import random

import tensorflow as tf
from gymnasium import Env
from keras import Input
from keras.layers import Dense
import numpy as np
import gymnasium


class ReplayBuffer:
    def __init__(self):
        self.size: int = 0
        self.current_index = 0
        self.state = None
        self.action = None
        self.new_state = None
        self.reward = None
        self.done = None
        self.batch_size = 128
        self.max_size = 4096

    def add_experience(self, state, action, reward, new_state, done):
        if self.size != 0:
            self.state = np.vstack((self.state, state))
            self.action = np.vstack((self.action, action))
            self.reward = np.vstack((self.reward, reward))
            self.new_state = np.vstack((self.new_state, new_state))
            self.done = np.vstack((self.done, done))
        else:
            if self.size < self.max_size:
                self.state = state
                self.action = np.array([action])
                self.reward = np.array([reward])
                self.new_state = new_state
                self.done = np.array([done])
            else:
                self.state[self.current_index % self.max_size, :] = state
                self.action[self.current_index % self.max_size, :] = action
                self.reward[self.current_index % self.max_size, :] = reward
                self.new_state[self.current_index % self.max_size, :] = new_state
                self.done[self.current_index % self.max_size, :] = done

        self.size += 1
        self.current_index += 1

    def sample_batch(self):
        batch_size = self.batch_size
        if self.batch_size > self.state.shape[0]:
            batch_size = self.size
        idx = np.random.randint(self.state.shape[0], size=batch_size)
        return self.state[idx], self.action[idx], self.reward[idx], self.new_state[idx], self.done[idx]

    def empty(self):
        self.size = 0
        self.current_index = 0


class DQNAgent:

    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount_factor = 0.95
        self.decay_rate = 0.995
        self.epsilon = 1
        self.learning_rate = 0.002
        self.replay_buffer = ReplayBuffer()
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.step_counter = 0

    def create_model(self):
        model = tf.keras.Sequential([
            Input(shape=(self.n_states,)),
            Dense(units=32, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()),
            Dense(units=64, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()),
            Dense(units=32, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()),
            Dense(units=16, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()),
            Dense(units=8, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform()),
            Dense(self.n_actions, activation='linear')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer, loss='mse', metrics=['mse'])
        return model

    def select_action(self, env: Env, state):
        if random.uniform(0, 1) < self.epsilon:
            return env.action_space.sample()
        else:
            q_values = self.model.predict(state, verbose=0)
            return np.argmax(q_values)

    def train_model(self):
        v_state, v_action, v_reward, v_new_state, v_done = self.replay_buffer.sample_batch()
        qsa = self.model.predict(v_state, verbose=0)
        qsa_target = self.target_model.predict(v_new_state, verbose=0)

        y_j = np.copy(qsa)
        y_j[np.arange(y_j.shape[0]), v_action.T] = v_reward.T + (v_done.T == 0) * self.discount_factor * np.max(
            qsa_target, axis=1)
        self.model.train_on_batch(v_state, y_j)

    def train(self, env: Env, max_episodes: int, max_steps):
        rewards = list()
        for episode in range(max_episodes):

            state, _ = env.reset()
            state = np.reshape(state, [1, self.n_states])
            rewards_per_episode = 0

            for step in range(max_steps):

                self.step_counter += 1
                # step function
                action = self.select_action(env, state)
                new_state, reward, done, info, _ = env.step(action)

                rewards_per_episode += reward

                new_state = np.reshape(new_state, [1, self.n_states])
                self.replay_buffer.add_experience(
                    state=state,
                    action=action,
                    new_state=new_state,
                    reward=reward,
                    done=done
                )

                self.train_model()
                state = new_state

                self.epsilon = 0.05 if self.epsilon * self.decay_rate < 0.05 else self.epsilon * self.decay_rate
                if self.step_counter % 100 == 0:
                    self.target_model.set_weights(self.model.weights)
                # end step function
                if done:
                    break

            rewards.append(rewards_per_episode)
            print(f"episode {episode} - rewards {rewards_per_episode} -epsilon {self.epsilon}")

            mean_rewards = np.array(rewards[-40:]).mean()
            if mean_rewards > 220:
                return rewards

        return rewards


def main():
    # create dqn
    env = gymnasium.make("CartPole-v1")
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    agent = DQNAgent(n_states, n_actions)

    rewards = agent.train(env, max_episodes=4096, max_steps=1024)
    np.savetxt('test.csv', np.array(rewards), delimiter=',')


#     num_actions = env.action_space.n
#     state_dim = env.observation_space.shape[0]

if __name__ == "__main__":
    main()

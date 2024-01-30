import random
import tensorflow as tf
from gymnasium import Env
from keras import Input
from keras.layers import Dense
import numpy as np
import gymnasium


class ExperienceBuffer:
    """
    Experience buffer class
    Operates as an experience buffer storing (state,action,reward,newstate,done) vectors to be replayed
    """

    def __init__(self, batch_size: int, max_size: int):
        self.size = 0
        self.current_index = 0
        self.v_state = None
        self.v_action = None
        self.v_new_state = None
        self.v_reward = None
        self.v_done = None
        self.batch_size = batch_size
        self.max_size = max_size

    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray,
                       done: bool) -> None:
        """
        Add experience to the buffer, if the addition exceeds the current size of the buffer remove the oldest instance
        :param state:
        :type state:
        :param action:
        :type action:
        :param reward:
        :type reward:
        :param new_state:
        :type new_state:
        :param done:
        :type done:
        :return:
        :rtype:
        """

        if self.size > self.max_size:
            self.v_state[self.current_index % self.max_size, :] = state
            self.v_action[self.current_index % self.max_size, :] = action
            self.v_reward[self.current_index % self.max_size, :] = reward
            self.v_new_state[self.current_index % self.max_size, :] = new_state
            self.v_done[self.current_index % self.max_size, :] = done
        else:
            if self.size != 0:
                self.v_state = np.vstack((self.v_state, state))
                self.v_action = np.vstack((self.v_action, action))
                self.v_reward = np.vstack((self.v_reward, reward))
                self.v_new_state = np.vstack((self.v_new_state, new_state))
                self.v_done = np.vstack((self.v_done, done))
            else:
                if self.size < self.max_size:
                    self.v_state = state
                    self.v_action = np.array([action])
                    self.v_reward = np.array([reward])
                    self.v_new_state = new_state
                    self.v_done = np.array([done])

            self.size += 1
        self.current_index = self.current_index % self.max_size + 1

    def sample_batch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Random sample a batch of at least size batch_size out of the experience buffer,
        :return:
        :rtype:
        """
        batch_size = self.batch_size if self.batch_size <= self.v_state.shape[0] else self.size
        idx = np.random.randint(self.v_state.shape[0], size=batch_size)
        return self.v_state[idx], self.v_action[idx], self.v_reward[idx], self.v_new_state[idx], self.v_done[idx]


class DQNAgent:
    """
    Operates the Deep q learning agent with competing target and q models
    """

    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount_factor = 0.95
        self.decay_rate = 0.995
        self.epsilon = 1
        self.learning_rate = 0.002
        self.replay_buffer = ExperienceBuffer(batch_size=128, max_size=4096)
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.step_counter = 0
        self.q_iteration = 128

    def create_model(self) -> tf.keras.Sequential:
        """
        Create a specified model
        :return:
        :rtype:
        """
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

    def to_stop(self, rewards: list, threshold: int) -> bool:
        """
        Return if rewards moving mean are above threshold
        :param rewards:
        :type rewards:
        :return:
        :rtype:
        """
        return np.array(rewards[-475:]).mean() > threshold

    def sample_action(self, env: Env, state: np.ndarray) -> np.ndarray:
        """
        Sample action with epsilon randomness
        :param env:
        :type env:
        :param state:
        :type state:
        :return:
        :rtype:
        """
        if random.uniform(0, 1) < self.epsilon:
            return env.action_space.sample()
        else:
            q_values = self.model.predict(state, verbose=0)
            return np.argmax(q_values)

    def train_model(self) -> int:
        """
        Train deep q learning model out of a batch taken from the replay buffer
        :return: loss of train model
        :rtype:
        """
        v_state, v_action, v_reward, v_new_state, v_done = self.replay_buffer.sample_batch()
        qsa = self.model.predict(v_state, verbose=0)
        qsa_target = self.target_model.predict(v_new_state, verbose=0)

        y_j = np.copy(qsa)
        y_j[np.arange(y_j.shape[0]), v_action.T] = v_reward.T + (v_done.T == 0) * self.discount_factor * np.max(
            qsa_target, axis=1)
        return self.model.train_on_batch(v_state, y_j)[0]

    def train(self, env: Env, max_episodes: int, max_steps: int):
        """
        Train q network with epsilon decay
        for each step in episode
        1) sample an action either by epsilon decay or model predict
        2) perform action on the environment and observe new state and rewards
        3) record experience to the experience buffer
        4) train model on random batch sampling
        5) decay epsilon
        6) for each q_iteration steps update target_model weight from model
        :param env:
        :type env:
        :param max_episodes:
        :type max_episodes:
        :param max_steps:
        :type max_steps:
        :return:
        :rtype:
        """
        rewards = list()
        loss = list()
        for episode in range(max_episodes):

            state, _ = env.reset()
            state = np.reshape(state, [1, self.n_states])
            rewards_per_episode = 0
            loss_per_step = 0
            for step in range(max_steps):

                self.step_counter += 1
                # step function
                action = self.sample_action(env, state)
                new_state, reward, done, info, _ = env.step(action)

                new_state = np.reshape(new_state, [1, self.n_states])
                self.replay_buffer.add_experience(
                    state=state,
                    action=action,
                    new_state=new_state,
                    reward=reward,
                    done=done
                )

                if self.step_counter % 10==0:
                    loss_per_step = self.train_model()

                else:
                    self.train_model()
                state = new_state
                rewards_per_episode += reward

                self.epsilon = 0.05 if self.epsilon * self.decay_rate < 0.05 else self.epsilon * self.decay_rate
                if self.step_counter % self.q_iteration == 0:
                    self.step_counter =0
                    self.target_model.set_weights(self.model.weights)
                # end step function
                if done:
                    break
            loss.append(loss_per_step)
            rewards.append(rewards_per_episode)
            print(f"episode {episode} - rewards {rewards_per_episode} -epsilon {self.epsilon}")

            if self.to_stop(rewards, 475):
                return rewards,loss

        return rewards,loss


def main():
    # create dqn
    env = gymnasium.make("CartPole-v1")
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    agent = DQNAgent(n_states, n_actions)

    rewards,losses = agent.train(env, max_episodes=4096, max_steps=1024)
    # np.savetxt('test.csv', np.array(rewards), delimiter=',')

    np.savetxt('rewards.csv', np.array(rewards), delimiter=',')
    np.savetxt('losses.csv', np.array(losses), delimiter=',')

#     num_actions = env.action_space.n
#     state_dim = env.observation_space.shape[0]

if __name__ == "__main__":
    main()

import tensorflow as tf
from gymnasium import Env
from keras import Input
from keras.layers import Dense
import numpy as np
import gymnasium


#
# # Define the Q-network model
# class QNetwork(tf.keras.Model):
#     def __init__(self, num_actions):
#         super(QNetwork, self).__init__()
#         self.dense1 = tf.keras.layers.Dense(64, activation='relu')
#         self.dense2 = tf.keras.layers.Dense(64, activation='relu')
#         self.output_layer = tf.keras.layers.Dense(num_actions, activation=None)
#
#     def call(self, state):
#         x = self.dense1(state)
#         x = self.dense2(x)
#         return self.output_layer(x)
#
# # Define the DQN agent
# class DQNAgent:
#     def __init__(self, num_actions, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
#         self.q_network = QNetwork(num_actions)
#         self.target_network = QNetwork(num_actions)
#         self.target_network.set_weights(self.q_network.get_weights())
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#         self.loss_function = tf.keras.losses.MeanSquaredError()
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_decay = epsilon_decay
#         self.epsilon_min = epsilon_min
#
#     def select_action(self, state):
#         if np.random.rand() < self.epsilon:
#             return np.random.randint(self.q_network.output_shape[-1])
#         q_values = self.q_network.predict(state)
#         return np.argmax(q_values)
#
#     def train(self, state, action, reward, next_state, done):
#         target = reward + (1 - done) * self.gamma * np.amax(self.target_network.predict(next_state), axis=1)
#         with tf.GradientTape() as tape:
#             q_values = self.q_network(state, training=True)
#             action_masks = tf.one_hot(action, self.q_network.output_shape[-1])
#             selected_q_values = tf.reduce_sum(action_masks * q_values, axis=1)
#             loss = self.loss_function(target, selected_q_values)
#         gradients = tape.gradient(loss, self.q_network.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
#
#     def update_target_network(self):
#         self.target_network.set_weights(self.q_network.get_weights())
#
#     def decay_epsilon(self):
#         self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
#
# # Main training loop
# def train_dqn(env_name, num_episodes=1000):
#     env = gym.make(env_name)
#     num_actions = env.action_space.n
#     state_dim = env.observation_space.shape[0]
#
#     agent = DQNAgent(num_actions)
#
#     for episode in range(num_episodes):
#         state,_ = env.reset()
#         state = np.reshape(state, [1, state_dim])
#
#         total_reward = 0
#         done = False
#
#         while not done:
#             action = agent.select_action(state)
#             next_state, reward, done, _ = env.step(action)
#             next_state = np.reshape(next_state, [1, state_dim])
#
#             agent.train(state, action, reward, next_state, done)
#             agent.update_target_network()
#
#             total_reward += reward
#             state = next_state
#
#         agent.decay_epsilon()
#
#         print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
#
#     env.close()
#
# # Example usage
#

class DQNAgent():

    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.model = tf.keras.Sequential([
            Input(shape=(n_states,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(16, activation='relu'),
            Dense(n_actions, activation='softmax')
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
        self.model.compile(optimizer,metrics=['accuracy'])

    def select_action(self, state):
        q_values = self.model.predict(state)
        return np.argmax(q_values)

    def train(self, env: Env, n_episodes):
        for episode in range(n_episodes):
            state,_ = env.reset()
            state = np.reshape(state,[1,self.n_states])

            done = False
            while not done:
                action = self.select_action(state)
                x=0
def main():
    # create dqn
    env = gymnasium.make("CartPole-v1")
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    agent = DQNAgent(n_states, n_actions)

    agent.train(env,5000)


#     num_actions = env.action_space.n
#     state_dim = env.observation_space.shape[0]

if __name__ == "__main__":
    main()

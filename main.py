'''
import gym
import random


def print_fxn():
    envs = gym.envs.registry.items()
    env_ids = [env_spec[0] for env_spec in envs]
    print(env_ids)

def main():

    env = gym.make('SpaceInvaders-v0')
    height, width, channels = env.observation_space.shape
    actions = env.action_space.n

if __name__ == '__main__':
    print_fxn()

'''

import gym
import numpy as np
from collections import deque
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import cv2


env = gym.make('ALE/Breakout-v5', render_mode='human')

# Preprocess function
def preprocess(frame):
    # Your preprocessing steps here (resizing, converting to grayscale, etc.)
    return processed_frame

# Define the DQN model
def build_model(input_shape, num_actions):
    model = Sequential([
        Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape),
        Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer=Adam(), loss='mse')
    return model

# Initialize DQN agent
state_shape = (84, 84, 4)  # Adjust according to your preprocessing
num_actions = env.action_space.n
agent = build_model(state_shape, num_actions)

# Define DQN agent
import random
from collections import deque
from tensorflow.keras.models import clone_model


class DQNAgent:
    def __init__(self, state_shape, num_actions, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Replay memory
        self.memory = deque(maxlen=100000)  # Adjust the maximum size of the replay memory

        # Initialize frames for state representation
        self.frames = deque(maxlen=4)  # Store 4 frames

        # Neural networks
        self.model = self._build_model()
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())


    def _build_model(self):
            model = Sequential([
                Dense(24, input_shape=self.state_shape, activation='relu'),
                Dense(24, activation='relu'),
                Dense(self.num_actions, activation='linear')
            ])
            model.compile(optimizer=Adam(), loss='mse')
            return model

    def preprocess_frame(self, frame):
        # Ensure frame is converted to a NumPy array
        frame = np.array(frame)

        # Convert to grayscale using RGB to BGR conversion
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize the frame to (84, 84)
        frame = cv2.resize(frame, (84, 84))

        # Normalize pixel values
        frame = frame.astype(np.float32) / 255.0
        return frame

    def stack_frames(self, state):
        processed_frame = self.preprocess_frame(state)
        if len(self.frames) == 0:
            # Initialize frames with the same processed frame
            self.frames = deque([processed_frame] * 4, maxlen=4)
        else:
            self.frames.append(processed_frame)

        # Stack frames along the last axis
        stacked_frames = np.stack(self.frames, axis=-1)

        # Ensure the shape matches the expected input shape
        stacked_frames = np.expand_dims(stacked_frames, axis=0)  # Add a batch dimension

        return stacked_frames

    def act(self, state):
        processed_state = self.stack_frames(state)
        q_values = self.model.predict(processed_state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        state = np.array(state[0])
        next_state = np.array(next_state)

        processed_state = self.stack_frames(state)
        processed_next_state = self.stack_frames(next_state)
        self.memory.append((processed_state, action, reward, processed_next_state, done))



    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = np.max(self.target_model.predict(next_state)[0])
                target[0][action] = reward + self.gamma * Q_future

            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Initialize DQN agent
state_shape = (84, 84, 4)  # Adjust according to your preprocessing
num_actions = env.action_space.n
agent = DQNAgent(state_shape, num_actions)

# Training loop
num_episodes = 1000
batch_size = 32

for episode in range(num_episodes):
    state = env.reset()
    # Preprocess the initial state

    done = False
    while not done:
        # Agent selects action
        action = agent.act(state)

        # Apply action to the environment
        next_state, reward, done, _ = env.step(action)[:4]
        # Preprocess the next state

        # Store experience in replay memory
        agent.remember(state, action, reward, next_state, done)

        # Train the agent
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        state = next_state


def test_agent():
    total_rewards = []
    for episode in range(10):
        state = env.reset()
        # Preprocess the initial state

        done = False
        episode_reward = 0
        while not done:
            # Choose the best action
            action = np.argmax(agent.predict(state.reshape((1, *state_shape))))

            # Apply action to the environment
            next_state, reward, done, _ = env.step(action)[:4]
            # Preprocess the next state

            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print("Average reward over 10 episodes:", avg_reward)


# Test the trained agent
test_agent()

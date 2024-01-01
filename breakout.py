import gym
import numpy as np
import random

# Create the Breakout environment
env = gym.make('ALE/Breakout-v5', render_mode='human', full_action_space=False)

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01  # Exponential decay rate for epsilon

# Get the action space and initialize Q-table
action_space_size = env.action_space.n
state_space_size = env.observation_space.shape[0]  # Update to get the size of the state space
q_table = np.zeros((state_space_size, action_space_size))

# Initialize the experience replay buffer
buffer_size = 10000
experience_replay = []

'''
# Function to add experiences to the replay buffer
def add_experience(state, action, reward, new_state, done):
    experience = (state, action, reward, new_state, done)
    experience_replay.append(experience)
    if len(experience_replay) > buffer_size:
        experience_replay.pop(0)  # Remove oldest experience if buffer is full

# Function to sample experiences from the replay buffer
def sample_experience(batch_size):
    if len(experience_replay) >= batch_size:
        return random.sample(experience_replay, batch_size)
    else:
        return experience_replay 
'''

# Function to choose an action based on epsilon-greedy strategy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore action space
    else:
        return np.argmax(q_table[state, :])



# Training the agent
total_episodes = 10000
max_steps = 100  # Maximum steps per episode

for episode in range(total_episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    for step in range(max_steps):
        action = choose_action(state)
        new_state, reward, done, _ = env.step(action)[:4]

        # Update Q-table using Bellman equation
        q_table[state, action] = q_table[state, action] + alpha * (
                    reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])

        total_reward += reward
        state = new_state

        if done:
            break

    # Decay epsilon to reduce exploration over time
    if done and total_reward < 100:  # Assuming reward < 100 indicates loss of life
        state = env.reset()
        done = False

        # Decay epsilon to reduce exploration over time
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}")

# Test the trained agent
total_episodes_test = 10
total_rewards_test = []

for episode in range(total_episodes_test):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(q_table[state, :])
        new_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = new_state

    total_rewards_test.append(total_reward)

# Print average reward over test episodes
print(f"Average Reward over {total_episodes_test} Test Episodes: {np.mean(total_rewards_test)}")

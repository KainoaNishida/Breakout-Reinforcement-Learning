import gym as gym

import time
import matplotlib.pyplot as plt

import keyboard

env=gym.make("ALE/Breakout-v5",render_mode="human",
             full_action_space=False,repeat_action_probability=0.1,obs_type='rgb')

env.reset()

num_actions = env.action_space.n

meaning = env.unwrapped.get_action_meanings()

actionDict={'w': 0, 's': 1, 'd': 2, 'a': 3}

obs, reward, terminated, truncated, info = env.step(1)

plt.figure(figsize=(8, 8))
plt.imshow(obs)

totalReward=0
while True:
    event = keyboard.read_event()

    if actionDict.get(event.name, -1) != -1:
        obs, reward, terminated, truncated, info = env.step(actionDict.get(event.name, -1))
    totalReward=totalReward+reward
    time.sleep(0.05)

env.reset()
env.close()


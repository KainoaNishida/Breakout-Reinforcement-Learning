# Reinforcement Learning with Deep-Q Learning for Atari Games

## Overview
Machine learning enthusiasts often venture into the captivating realm of reinforcement learning, seeking to master classic Atari games. This project aims to craft powerful agents proficient in playing and excelling at these cherished games.

## About Deep-Q Learning
Deep-Q Learning is a technique in reinforcement learning, merging deep learning and Q-learning. It empowers agents to make sequential decisions in an environment by estimating the quality of actions through a function that predicts future rewards for each possible action in a given state.

## Setup Instructions

### 1. Set Up Virtual Environment
Create and activate a virtual environment to ensure proper isolation of dependencies:

```bash
# Create a virtual environment
python -m venv myenv

# Activate the virtual environment (Windows)
myenv\Scripts\activate

# Activate the virtual environment (macOS/Linux)
source myenv/bin/activate
```

### 2. Install Dependencies
Use pip to install the necessary packages:

```bash
pip install gym
pip install numpy
pip install gym[atari]
pip install atari-py
pip install gym[classic_control]
pip install tensorflow
pip install torch torchvision
pip install matplotlib
```

### 3. Download Project Files
Clone or download all required files from the repository

### 4. Run the Program
Execute the following command:

```bash
python -m [game title]
```

# ML 2024 Mini Project 4

Welcome to the Mini Project 4 repository. This project focuses on training and evaluating Deep Q-Network (DQN) and Double Deep Q-Network (DDQN) agents on the Lunar Lander environment using different batch sizes.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [DQN & DDQN Training](#q1-dqn-training)


## Overview

This project is divided into two main parts, each focusing on training DQN and DDQN agents with different batch sizes:
1. Training and evaluating DQN agents.
2. Training and evaluating DDQN agents.

For more detailed information, please refer to the accompanying report file.

## Requirements

Ensure you have the following installed:
- Python 3.x
- Jupyter Notebook
- Required Python packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `gym`, `stable-baselines3`, `pyvirtualdisplay`, `box2d-py`

Install the required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow gym stable-baselines3 pyvirtualdisplay box2d-py
```

### Part 1: DQN Training

#### Section 1: Setup and Environment

##### Tasks:

- Installing necessary packages and setting up the environment.
- Initializing the Lunar Lander environment.
- Defining the DQN architecture and experience replay mechanism.

#### Section 2: Training DQN Agents

##### Tasks:

- Training DQN agents with batch sizes 32, 64, and 128.
- Saving the trained models and recording videos of the agents.

#### Section 3: Evaluating DQN Agents

##### Tasks:

- Evaluating the performance of DQN agents on the Lunar Lander environment.
- Plotting the rewards over episodes and saving the results.

### Part 2: DDQN Training

#### Section 1: Setup and Environment

##### Tasks:

- Installing necessary packages and setting up the environment.
- Initializing the Lunar Lander environment.
- Defining the DDQN architecture and experience replay mechanism.

#### Section 2: Training DDQN Agents

##### Tasks:

- Training DDQN agents with batch sizes 64 and 128.
- Saving the trained models and recording videos of the agents.

#### Section 3: Evaluating DDQN Agents

##### Tasks:

- Evaluating the performance of DDQN agents on the Lunar Lander environment.
- Plotting the rewards over episodes and saving the results.

The results and video of the performance of the agent are located in the result folder.
For more detailed information on the implementation, results, and analysis, please refer to the accompanying report file.

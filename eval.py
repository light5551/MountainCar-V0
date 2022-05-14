import random
import torch
import torch.nn as nn
import numpy as np
import gym
from torch.utils.tensorboard import SummaryWriter
#from train import crate_new_model

episode_length = 500
env = gym.make("MountainCar-v0")
env = env.unwrapped
model = nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 3)
    )
model.load_state_dict(torch.load('model.pth'))
state = env.reset()                                                     # Reset environment
episode_reward_sum = 0                                              # Initialize the total reward of episode corresponding to this cycle
for i in range(episode_length):                                                         # Start an episode (each cycle represents a step)
    env.render()                                                    # Show experimental animation
    output = model.forward(torch.Tensor(state)).detach()  # In output is [left walking cumulative reward, right walking cumulative reward]
    print(torch.argmax(output).data.item())
    action = torch.argmax(output).data.item()  # Select action with argmax

    state_, reward, done, info = env.step(action)  # Perform actions to get feedback from env

import gym
import torch

from model import get_model

MODEL_PATH = 'model.pth'
TARGET_MODEL_PATH = 'target_model.pth'

episode_length = 500
env = gym.make("MountainCar-v0")
env = env.unwrapped
model = get_model()
model.load_state_dict(torch.load(MODEL_PATH))
state = env.reset()  # Reset environment
episode_reward_sum = 0  # Initialize the total reward of episode corresponding to this cycle
for i in range(episode_length):  # Start an episode (each cycle represents a step)
    env.render()  # Show experimental animation
    output = model.forward(
        torch.Tensor(state)).detach()  # In output is [left walking cumulative reward, right walking cumulative reward]
    # print(output)
    # print(torch.argmax(output))
    # print(torch.argmax(output).data.item())
    action = torch.argmax(output).data.item()  # Select action with argmax

    state_, reward, done, info = env.step(action)  # Perform actions to get feedback from env
    state = state_

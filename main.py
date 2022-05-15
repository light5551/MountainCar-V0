import gym
import pygame

from joystick import update_joystick

env = gym.make("MountainCar-v0")
env.action_space.seed(42)

observation, info = env.reset(seed=42, return_info=True)
stack = [1]  # accelerate


def update():

    action = update_joystick()
    print(action)
    observation, reward, done, info = env.step(action)
    env.render()
    if done:
        observation, info = env.reset(return_info=True)


env.render()
for _ in range(1000):
    update()

print('quit')

env.close()

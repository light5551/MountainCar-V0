import gym
import numpy as np
import pygame

env = gym.make("MountainCar-v0")
env.action_space.seed(42)

observation, info = env.reset(seed=42, return_info=True)
stack = [1]  # accelerate


def update():
    events = pygame.event.get()

    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                stack.append(0)
            if event.key == pygame.K_RIGHT:
                stack.append(2)
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                stack.remove(0)
            if event.key == pygame.K_RIGHT:
                stack.remove(2)

    action = stack[-1]
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

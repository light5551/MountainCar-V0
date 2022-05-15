import pygame

stack = [1]  # accelerate

'''
| Num | Observation                                                 | Value   |     Unit     |
|-----|-------------------------------------------------------------|---------|--------------|
| 0   | Accelerate to the left                                      | Inf     | position (m) |
| 1   | Don't accelerate                                            | Inf     | position (m) |
| 2   | Accelerate to the right                                     | Inf     | position (m) |
'''


def update_joystick() -> int:
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

    return stack[-1]


if __name__ == '__main__':
    import gym

    env = gym.make("MountainCar-v0")
    env.action_space.seed(42)

    observation, info = env.reset(seed=42, return_info=True)


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

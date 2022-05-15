import gym
import itertools


class ResetCounter(gym.Wrapper):

    def __init__(self, env):
        super(ResetCounter, self).__init__(env)
        self.__counter = itertools.count(-1)
        self.count = 0

    def reset(self, **kwargs):
        self.count = next(self.__counter)
        return self.env.reset(**kwargs)


def get_env():
    env = gym.make("MountainCar-v0")
    env = ResetCounter(env)
    return env

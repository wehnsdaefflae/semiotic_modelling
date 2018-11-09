# coding=utf-8

# TODO: implement!
# https://pythonprogramming.net/openai-cartpole-neural-network-example-machine-learning-tutorial/
import time

import gym
import numpy

from _framework.systems.controllers.rational.implementations.rational_sarsa import RationalSarsa
from tools.functionality import smear

gym.envs.register(
    id="MountainCar-infinite-v0",
    entry_point="_framework.systems.tasks.rational.resources.infinite_mountain_car:ContinuousMountainCarEnv",
    max_episode_steps=10,       # ignore?
    reward_threshold=-110.0,    # ignore?
)


def rational():
    # https://github.com/openai/gym/blob/master/gym/envs/__init__.py
    env = gym.make("MountainCar-infinite-v0")
    env.reset()

    controller = RationalSarsa(((-1., 1.),), 4, 100, .5, .25)

    def some_random_games_first():
        average_reward = 0.
        iterations = 0
        sensor = None
        while True:
            env.render()

            if True or sensor is None:
                motor = env.action_space.sample()
                # motor = numpy.array((0.,))
            else:
                motor = numpy.array(controller.react(sensor))

            state = env.step(motor)
            sensor, reward, done, info = state

            # (x_pos, x_vel, theta_ang, theta_vel)

            # controller.integrate(sensor, tuple(motor), reward)

            average_reward = smear(average_reward, reward, iterations)
            iterations += 1

            if iterations % 100 == 0:
                print(state)
                print(average_reward)

    some_random_games_first()
    env.close()


if __name__ == "__main__":
    rational()

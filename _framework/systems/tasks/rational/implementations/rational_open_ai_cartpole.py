# coding=utf-8

# TODO: implement!
# https://pythonprogramming.net/openai-cartpole-neural-network-example-machine-learning-tutorial/
import time

import gym
import numpy

from _framework.systems.controllers.nominal.implementations.nominal_sarsa_controller import NominalSarsaController
from _framework.systems.controllers.rational.implementations.rational_sarsa import RationalSarsa
from tools.functionality import smear

gym.envs.register(
    id="CartPole-infinite-v0",
    entry_point="_framework.systems.tasks.rational.resources.infinite_cartpole:InfiniteCartPoleEnv",
    max_episode_steps=10,       # ignore?
    reward_threshold=-110.0,    # ignore?
)


def rational():

    # https://github.com/openai/gym/blob/master/gym/envs/__init__.py
    env = gym.make("CartPole-infinite-v0")
    env.reset()

    controller = RationalSarsa(((-1., 1.),), 4, 100, .5, .25)
    # controller = NominalSarsaController(("l", "r"), .1, .5, .1)

    def some_random_games_first():
        # Each of these is its own game.
        # this is each frame, up to 200...but we wont make it that far.

        average_reward = 0.
        iterations = 0
        sensor = None
        while True:
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right

            if sensor is None:
                # motor = env.action_space.sample()
                motor = numpy.array((0.,))
            else:
                motor = numpy.array(controller.react(sensor))

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            state = env.step(motor)
            sensor, reward, done, info = state

            # (x_pos, x_vel, theta_ang, theta_vel)

            controller.integrate(sensor, tuple(motor), reward)

            average_reward = smear(average_reward, reward, iterations)
            iterations += 1

            if iterations % 1000 == 0:
                print(average_reward)

    some_random_games_first()
    env.close()


def nominal():

    env = gym.make("CartPole-infinite-v0")
    env.reset()

    controller = NominalSarsaController(("l", "r"), .1, .5, .1)

    def some_random_games_first():
        average_reward = 0.
        iterations = 0
        sensor = None
        while True:
            #env.render()

            if sensor is None:
                # motor = env.action_space.sample()
                motor = "l"
            else:
                motor = numpy.array(controller.react(sensor))

            state = env.step(motor)
            (x_pos, x_vel, theta_ang, theta_vel), reward, done, info = state

            controller.integrate(sensor, tuple(motor), reward)

            average_reward = smear(average_reward, reward, iterations)
            iterations += 1

            if iterations % 1000 == 0:
                print(average_reward)

    some_random_games_first()
    env.close()


if __name__ == "__main__":
    rational()

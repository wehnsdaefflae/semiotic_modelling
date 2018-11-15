# coding=utf-8

# TODO: implement!
# https://pythonprogramming.net/openai-cartpole-neural-network-example-machine-learning-tutorial/
import math
import os
from collections import deque

import gym
import numpy
from matplotlib import pyplot

from _framework.systems.controllers.rational.implementations.rational_sarsa import RationalSarsa
from _framework.systems.tasks.rational.resources.custom_infinite_mountain_car import MountainCar
from tools.functionality import smear, clip
from tools.regression import plot_surface
from tools.timer import Timer

gym.envs.register(
    id="VanillaMountainCar-infinite-v0",
    entry_point="_framework.systems.tasks.rational.resources.vanilla_infinite_mountain_car:ContinuousMountainCarEnv",
    max_episode_steps=10,       # ignore?
    reward_threshold=-110.0,    # ignore?
)

gym.envs.register(
    id="CustomMountainCar-infinite-v0",
    entry_point="_framework.systems.tasks.rational.resources.custom_infinite_mountain_car:MountainCar",
    max_episode_steps=10,       # ignore?
    reward_threshold=-110.0,    # ignore?
)


def basic():
    mc = MountainCar()
    # sc = SarsaController()
    while True:
        # motor = random.uniform(-1., 1.)
        motor = -1.
        sensor, reward = mc.respond(motor)
        # print(sensor)
        mc.render()
        #time.sleep(.1)

    mc.close()


def setup_3d_axis():
    from mpl_toolkits.mplot3d import Axes3D

    fig = pyplot.figure()
    plot_axis = fig.add_subplot(111, projection="3d")
    # plot_axis.set_aspect("equal")
    plot_axis.set_xlabel("location")
    plot_axis.set_ylabel("velocity")
    plot_axis.set_zlabel("action")
    plot_axis.set_zlim((-1., 1.))

    return plot_axis


def rational():
    # https://github.com/openai/gym/blob/master/gym/envs/__init__.py
    env = gym.make("CustomMountainCar-infinite-v0")
    # env = gym.make("VanillaMountainCar-infinite-v0")
    env.reset()

    controller = RationalSarsa(((-1., 1.),), 2, 100, .9, .002, polynomial_degree=3)
    # controller = NominalSarsaController(("l", "r"), .1, .5, .1)

    def some_random_games_first():
        # Each of these is its own game.
        # this is each frame, up to 200...but we wont make it that far.

        average_reward = 0.
        iterations = 0
        sensor = None
        visualize = False

        sensor_range = (-math.pi, math.pi), (-.02, .02)
        axis = setup_3d_axis()
        last_points = deque(maxlen=100)

        policy = None
        scatter = None

        while True:
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            if average_reward >= .9:
                visualize = True
            if visualize:
                env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right

            if sensor is None:
                # motor = env.action_space.sample()
                motor = .0,
            else:
                motor = controller.react(tuple(sensor))

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            state = env.step(numpy.array(motor))
            sensor, reward, done, info = state

            # (x_loc, x_vel)

            controller.integrate(tuple(sensor), tuple(motor), reward)

            last_points.append((*sensor, motor))

            average_reward = smear(average_reward, reward, iterations)
            iterations += 1

            if Timer.time_passed(2000):
                print(f"{iterations:010d} iterations, average reward: {average_reward:.2f}")
                print(sensor)
                if policy is not None:
                    policy.remove()
                if scatter is not None:
                    scatter.remove()
                policy = plot_surface(axis, lambda _x, _y: controller._decide((_x, _y))[0], sensor_range)
                scatter = axis.scatter(*zip(*last_points))
                pyplot.draw()
                pyplot.pause(.001)
                # controller.save_as("controller.sys")

    some_random_games_first()
    env.close()


if __name__ == "__main__":
    rational()

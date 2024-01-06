# coding=utf-8

# TODO: implement!
# https://pythonprogramming.net/openai-cartpole-neural-network-example-machine-learning-tutorial/
from collections import deque

import gym
import numpy
from matplotlib import pyplot

from _framework.systems.controllers.rational.implementations.actor_critic import RationalSarsa as RationalSarsaAC
from _framework.systems.controllers.rational.implementations.advantage import RationalSarsa as RationalSarsaADV
from _framework.systems.controllers.rational.implementations.random_controller import RandomRational
from _framework.systems.tasks.rational.resources.custom_infinite_mountain_car import MountainCar
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


def actor_critic():
    fig = pyplot.figure()
    actor_axis, critic_axis, reward_axis = fig.subplots(nrows=3, sharex="all")

    actor_axis.set_ylabel("actor error")
    actor_axis.yaxis.set_label_position("right")

    critic_axis.set_ylabel("critic error")
    critic_axis.yaxis.set_label_position("right")

    reward_axis.set_ylabel("average reward")
    reward_axis.yaxis.set_label_position("right")
    reward_axis.set_xlabel("iteration")

    # https://github.com/openai/gym/blob/master/gym/envs/__init__.py
    env = gym.make("CustomMountainCar-infinite-v0")
    # env = gym.make("VanillaMountainCar-infinite-v0")
    env.reset()

    controller = RationalSarsaAC(((-.5, .5),), 2, 500, 5, .5, .1, polynomial_degree=2)
    # controller = RandomRational(((-.5, .5),))

    sensor = None
    visualize = False
    plot = True

    window_size = 100000

    actor_plot, critic_plot, reward_plot = None, None, None
    actor_data, critic_data, reward_data = deque(maxlen=window_size), deque(maxlen=window_size), deque(maxlen=window_size)

    while True:
        if visualize:
            env.render()

        if sensor is None:
            # motor = tuple(random.uniform(*_range) for _range in MountainCar.motor_range())
            motor = .0,
        else:
            motor = controller.react(tuple(sensor))

        state = env.step(numpy.array(motor))
        sensor, reward, done, info = state
        # (x_loc, x_vel)

        controller.integrate(tuple(sensor), tuple(motor), reward)

        actor_data.append(controller.average_actor_error)
        critic_data.append(controller.average_critic_error)
        reward_data.append(controller.average_reward)

        if Timer.time_passed(500) and plot:
            if actor_plot is not None:
                actor_plot.remove()
            if critic_plot is not None:
                critic_plot.remove()
            if reward_plot is not None:
                reward_plot.remove()

            x_min = max(controller.get_iterations() - window_size, 0)

            actor_plot, = actor_axis.plot(range(x_min, x_min + len(actor_data)), actor_data, color="black")
            actor_axis.set_xlim((x_min, x_min + window_size))
            actor_axis.set_ylim((min(actor_data), max(actor_data)))

            critic_plot, = critic_axis.plot(range(x_min, x_min + len(critic_data)), critic_data, color="black")
            critic_axis.set_xlim((x_min, x_min + window_size))
            critic_axis.set_ylim((min(critic_data), max(critic_data)))

            reward_plot, = reward_axis.plot(range(x_min, x_min + len(reward_data)), reward_data, color="black")
            reward_axis.set_xlim((x_min, x_min + window_size))
            reward_axis.set_ylim((min(reward_data), max(reward_data)))

            pyplot.tight_layout()
            pyplot.draw()
            pyplot.pause(.001)

    env.close()


def advantage():
    fig = pyplot.figure()
    actor_axis, advantage_axis, value_axis, reward_axis = fig.subplots(nrows=4, sharex="all")

    actor_axis.set_ylabel("actor error")
    actor_axis.yaxis.set_label_position("right")

    advantage_axis.set_ylabel("advantage error")
    advantage_axis.yaxis.set_label_position("right")

    value_axis.set_ylabel("value error")
    value_axis.yaxis.set_label_position("right")

    reward_axis.set_ylabel("average reward")
    reward_axis.yaxis.set_label_position("right")
    reward_axis.set_xlabel("iteration")

    # https://github.com/openai/gym/blob/master/gym/envs/__init__.py
    env = gym.make("CustomMountainCar-infinite-v0")
    # env = gym.make("VanillaMountainCar-infinite-v0")
    env.reset()

    controller = RationalSarsaADV(((-.5, .5),), 2, 10000, 5, .5, .1, polynomial_degree=2)
    # controller = RandomRational(((-.5, .5),))

    sensor = None
    visualize = False
    plot = True

    window_size = 100000

    actor_plot, advantage_plot, value_plot, reward_plot = None, None, None, None
    actor_data, advantage_data, value_data, reward_data = deque(maxlen=window_size), deque(maxlen=window_size), deque(maxlen=window_size), deque(maxlen=window_size)

    while True:
        if visualize:
            env.render()

        if sensor is None:
            # motor = tuple(random.uniform(*_range) for _range in MountainCar.motor_range())
            motor = .0,
        else:
            motor = controller.react(tuple(sensor))

        state = env.step(numpy.array(motor))
        sensor, reward, done, info = state
        # (x_loc, x_vel)

        controller.integrate(tuple(sensor), tuple(motor), reward)

        actor_data.append(controller.average_actor_error)
        advantage_data.append(controller.average_advantage_error)
        value_data.append(controller.average_value_error)
        reward_data.append(controller.average_reward)

        if Timer.time_passed(500) and plot:
            if actor_plot is not None:
                actor_plot.remove()
            if advantage_plot is not None:
                advantage_plot.remove()
            if value_plot is not None:
                value_plot.remove()
            if reward_plot is not None:
                reward_plot.remove()

            x_min = max(controller.get_iterations() - window_size, 0)

            actor_plot, = actor_axis.plot(range(x_min, x_min + len(actor_data)), actor_data, color="black")
            actor_axis.set_xlim((x_min, x_min + window_size))
            actor_axis.set_ylim((min(actor_data), max(actor_data)))

            advantage_plot, = advantage_axis.plot(range(x_min, x_min + len(advantage_data)), advantage_data, color="black")
            advantage_axis.set_xlim((x_min, x_min + window_size))
            advantage_axis.set_ylim((min(advantage_data), max(advantage_data)))

            value_plot, = value_axis.plot(range(x_min, x_min + len(value_data)), value_data, color="black")
            value_axis.set_xlim((x_min, x_min + window_size))
            value_axis.set_ylim((min(value_data), max(value_data)))

            reward_plot, = reward_axis.plot(range(x_min, x_min + len(reward_data)), reward_data, color="black")
            reward_axis.set_xlim((x_min, x_min + window_size))
            reward_axis.set_ylim((min(reward_data), max(reward_data)))

            pyplot.tight_layout()
            pyplot.draw()
            pyplot.pause(.001)

    env.close()


if __name__ == "__main__":
    actor_critic()
    # advantage()

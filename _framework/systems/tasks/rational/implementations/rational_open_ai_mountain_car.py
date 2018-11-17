# coding=utf-8

# TODO: implement!
# https://pythonprogramming.net/openai-cartpole-neural-network-example-machine-learning-tutorial/
from collections import deque

import gym
import numpy
from matplotlib import pyplot

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


def setup_axes():
    # from mpl_toolkits.mplot3d import Axes3D

    fig = pyplot.figure()

    """
    plot_axis = fig.add_subplot(211, projection="3d")
    plot_axis.set_xlabel("location")
    plot_axis.set_ylabel("velocity")
    plot_axis.set_zlabel("action")
    plot_axis.set_zlim((-1., 1.))
    """

    actor_axis, critic_axis, reward_axis = fig.subplots(nrows=3, sharex="all")

    actor_axis.set_ylabel("actor error")
    actor_axis.yaxis.set_label_position("right")

    critic_axis.set_ylabel("critic error")
    critic_axis.yaxis.set_label_position("right")

    reward_axis.set_ylabel("average reward")
    reward_axis.yaxis.set_label_position("right")
    reward_axis.set_xlabel("iteration")

    return actor_axis, critic_axis, reward_axis


def rational():
    # https://github.com/openai/gym/blob/master/gym/envs/__init__.py
    env = gym.make("CustomMountainCar-infinite-v0")
    # env = gym.make("VanillaMountainCar-infinite-v0")
    env.reset()

    # controller = RationalSarsa(((-.5, .5),), 2, 500, 5, .5, .1, polynomial_degree=2)
    controller = RandomRational(((-.5, .5),))

    def some_random_games_first():
        # Each of these is its own game.
        # this is each frame, up to 200...but we wont make it that far.

        sensor = None
        visualize = True
        plot = False

        window_size = 100000

        actor_axis, critic_axis, reward_axis = setup_axes()
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
            """
            actor_data.append(controller.average_actor_error)
            critic_data.append(controller.average_critic_error)
            """
            reward_data.append(controller.average_reward)

            if Timer.time_passed(500) and plot:
                if actor_plot is not None:
                    actor_plot.remove()
                if critic_plot is not None:
                    critic_plot.remove()
                if reward_plot is not None:
                    reward_plot.remove()

                x_min = max(controller._iteration - window_size, 0)
                x_max = x_min + min(window_size, controller._iteration)

                actor_plot, = actor_axis.plot(range(x_min, x_max), actor_data, color="black")
                actor_axis.set_xlim((x_min, x_min + window_size))
                actor_axis.set_ylim((min(actor_data), max(actor_data)))

                critic_plot, = critic_axis.plot(range(x_min, x_max), critic_data, color="black")
                critic_axis.set_xlim((x_min, x_min + window_size))
                critic_axis.set_ylim((min(critic_data), max(critic_data)))

                reward_plot, = reward_axis.plot(range(x_min, x_max), reward_data, color="black")
                reward_axis.set_xlim((x_min, x_min + window_size))
                reward_axis.set_ylim((min(reward_data), max(reward_data)))

                pyplot.tight_layout()
                pyplot.draw()
                pyplot.pause(.001)

    some_random_games_first()
    env.close()


if __name__ == "__main__":
    rational()

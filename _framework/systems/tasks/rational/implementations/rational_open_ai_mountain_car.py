# coding=utf-8

# TODO: implement!
# https://pythonprogramming.net/openai-cartpole-neural-network-example-machine-learning-tutorial/
import math
import random
import time
from math import cos
from typing import Tuple, Optional

import gym
import numpy

from _framework.data_types import RATIONAL_MOTOR, RATIONAL_SENSOR
from _framework.systems.controllers.rational.implementations.rational_sarsa import RationalSarsa
from _framework.systems.tasks.rational.abstract import RationalTask
from data_generation.data_sources.systems.controller_nominal import SarsaController
from tools.functionality import smear, signum

gym.envs.register(
    id="MountainCar-infinite-v0",
    entry_point="_framework.systems.tasks.rational.resources.infinite_mountain_car:ContinuousMountainCarEnv",
    max_episode_steps=10,       # ignore?
    reward_threshold=-110.0,    # ignore?
)


class MountainCar(RationalTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mass = 100.
        self._at_top = False
        self._location = 0.
        self._velocity = 0.
        self._hill = lambda _x: (-cos(_x) + 1.) / 2.
        self._hill_force = lambda _x: -signum(_x) * (-cos(_x * 2.) + 1.) / 2.

    def react(self, data_in: Optional[RATIONAL_MOTOR]) -> RATIONAL_SENSOR:
        force = data_in * .5 + self._hill_force(self._location)
        acceleration = force / self._mass
        self._velocity += acceleration
        self._location += self._velocity

        if self._location >= math.pi:
            self._at_top = True
            self._location = 0.
            self._velocity = 0.

        elif -math.pi >= self._location:
            self._at_top = True
            self._location = 0.
            self._velocity = 0.

        elif self._at_top:
            self._at_top = False

        return self._location, self._velocity

    @staticmethod
    def motor_range() -> Tuple[Tuple[float, float], ...]:
        return (-1., 1.),

    def _get_height(self, location: float) -> float:
        return self._hill(location)

    def _get_reward(self) -> float:
        return 10. if self._at_top else -1.


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
    mc = MountainCar()
    sc = SarsaController()
    while True:
        # motor = random.uniform(-1., 1.)
        motor = -1.
        print(mc.respond(motor))
        time.sleep(.1)

    # rational()

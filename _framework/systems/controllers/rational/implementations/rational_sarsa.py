# coding=utf-8
from typing import Tuple

from _framework.data_types import RATIONAL_SENSOR, RATIONAL_MOTOR
from _framework.systems.controllers.rational.abstract import RationalController
from tools.functionality import signum, clip
from tools.regression import MultiplePolynomialFromLinearRegression, MultivariatePolynomialRegression


class RationalSarsa(RationalController):
    def __init__(self,
                 motor_range: Tuple[Tuple[float, float], ...], sensor_dimensionality: int,
                 drag: int, gamma: float, epsilon: float, polynomial_degree: int = 1):
        super().__init__(motor_range, epsilon)
        self._drag = drag
        self._gamma = gamma

        self._last_sensor = None
        self._last_motor = None
        self._last_reward = 0.
        self._last_condition = None

        self._sensor_dim = sensor_dimensionality

        self._critic_input_dim = len(motor_range) + sensor_dimensionality
        self._motor_range = motor_range

        self._critic = MultiplePolynomialFromLinearRegression(sensor_dimensionality + len(motor_range), polynomial_degree, drag=100)    # S x M -> float
        self._actor = MultivariatePolynomialRegression(sensor_dimensionality, len(motor_range), polynomial_degree, drag=100)            # S -> M

        self._iteration = 0

    def _decide(self, sensor: RATIONAL_SENSOR) -> RATIONAL_MOTOR:
        action = tuple(clip(_m, *_ranges) for _m, _ranges in zip(self._actor.output(sensor), self._motor_range))
        return action

    def _integrate(self, sensor: RATIONAL_SENSOR, motor: RATIONAL_MOTOR, reward: float):
        # https://mpatacchiola.github.io/blog/2017/02/11/dissecting-reinforcement-learning-4.html
        if self._iteration >= 1:
            evaluation = self._critic.output(sensor + motor)
            last_eval = self._last_reward + self._gamma * evaluation

            self._critic.fit(self._last_sensor + self._last_motor, last_eval, drag=self._drag)

            best_known = tuple(clip(_m, *_ranges) for _m, _ranges in zip(self._actor.output(self._last_sensor), self._motor_range))
            best_known_eval = self._critic.output(self._last_sensor + best_known)

            delta_eval = last_eval - best_known_eval
            delta_step = tuple(_l - _b for _l, _b in zip(self._last_motor, best_known))        # TODO: test fixed step size
            better_motor = tuple(clip(_b + _d * delta_eval * .01, *_ranges) for _b, _d, _ranges in zip(best_known, delta_step, self._motor_range))

            self._actor.fit(self._last_sensor, better_motor, drag=self._drag)

        self._last_sensor, self._last_motor = sensor, motor
        self._last_reward = reward

        self._iteration += 1

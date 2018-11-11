# coding=utf-8
from typing import Tuple

from _framework.data_types import RATIONAL_SENSOR, RATIONAL_MOTOR
from _framework.systems.controllers.rational.abstract import RationalController
from tools.regression_experiments import MultiplePolynomialRegressor, FullPolynomialRegressor


class RationalSarsa(RationalController):
    def __init__(self,
                 motor_range: Tuple[Tuple[float, float], ...], sensor_dimensionality: int,
                 alpha: int, gamma: float, epsilon: float, polynomial_degree: int = 1):
        super().__init__(motor_range, epsilon)
        self._alpha = alpha
        self._gamma = gamma

        self._last_sensor = None
        self._last_motor = None
        self._last_reward = 0.
        self._last_condition = None

        self._sensor_dim = sensor_dimensionality

        self._critic_input_dim = len(motor_range) + sensor_dimensionality

        self._critic = MultiplePolynomialRegressor([polynomial_degree for _ in range(len(motor_range) + sensor_dimensionality)])    # approximate S x M -> float
        self._actor = FullPolynomialRegressor([polynomial_degree for _ in range(sensor_dimensionality)], len(motor_range))          # approximate S -> M

        self._iteration = 0

    def _decide(self, sensor: RATIONAL_SENSOR) -> RATIONAL_MOTOR:
        return self._actor.output(sensor)

    def _integrate(self, sensor: RATIONAL_SENSOR, motor: RATIONAL_MOTOR, reward: float):
        if self._iteration >= 1:
            evaluation = self._critic.output(sensor + motor)
            last_eval = self._last_reward + self._gamma * evaluation

            self._critic.fit(self._last_sensor + self._last_motor, last_eval, self._alpha)

            best_known = self._actor.output(self._last_sensor)
            best_known_eval = self._critic.output(self._last_sensor + best_known)

            delta_eval = last_eval - best_known_eval
            delta_step = tuple((_l - _b) * delta_eval for _l, _b in zip(self._last_motor, best_known))
            better_motor = tuple(_b + _d for _b, _d in zip(best_known, delta_step))

            self._actor.fit(self._last_sensor, better_motor, self._alpha)

        self._last_sensor, self._last_motor = sensor, motor
        self._last_reward = reward

        self._iteration += 1

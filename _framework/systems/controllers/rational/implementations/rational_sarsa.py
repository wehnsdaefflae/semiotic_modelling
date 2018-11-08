# coding=utf-8
import random
from typing import Tuple

from _framework.data_types import RATIONAL_SENSOR, RATIONAL_MOTOR
from _framework.systems.controllers.rational.abstract import RationalController
from tools.regression import MultiplePolynomialRegressor, FullPolynomialRegressor


class RationalSarsa(RationalController):
    def __init__(self,
                 motor_range: Tuple[Tuple[float, float], ...], sensor_range: Tuple[Tuple[float, float], ...],
                 alpha: int, gamma: float, mean: float, deviation: float):
        super().__init__(motor_range, mean, deviation)
        self._alpha = alpha
        self._gamma = gamma

        self._sensor = None
        self._motor = None
        self._reward = 0.
        self._last_condition = None

        # TODO: choose appropriate predictors
        self._critic = None     # approximate S x M -> float    # RationalPredictor (alternative input: state of best action predictor + sensor)
        self._actor = None      # approximate S -> M            # RationalPredictor (alternative input: state of evaluation predictor + sensor)
                                                                # alternative output motor and best evaluation
        self._sensor_range = sensor_range

        self._degree = 3
        self._critic_input_dim = len(motor_range) + len(sensor_range)

        # TODO: how to do?
        self._critic = dict()
        self._current_critic = None
        # self._critic = MultiplePolynomialRegressor([3 for _ in range(len(motor_range) + len(sensor_range))])
        self._actor = FullPolynomialRegressor([3 for _ in range(len(sensor_range))], len(motor_range))

        self._iteration = 0

    def _decide(self, sensor: RATIONAL_SENSOR) -> RATIONAL_MOTOR:
        return self._actor.output(sensor)

    def _integrate(self, sensor: RATIONAL_SENSOR, motor: RATIONAL_MOTOR, reward: float):
        if self._iteration >= 1:
            this_condition = sensor + motor
            evaluation = self._critic.output(this_condition)

            update_value = self._reward + self._gamma * evaluation

            last_condition = self._last_sensor, self._last_motor
            self._critic.fit(last_condition, update_value, self._alpha)

            # TODO: make MultiplePolynomialRegressor into Predictor
            # _last_perception = self._critic.get_state() + self._last_sensor
            _last_perception = tuple() + self._last_sensor

            if self._iteration >= 2:
                best_last_action = self._actor.output(_last_perception)       # _last_perception
                best_eval = self._critic(_last_perception, best_last_action)

                if best_eval < update_value:
                    self._actor.fit(self._last_sensor, self._last_motor)        # _last_perception

            self._last_sensor, self._last_motor = self._sensor, self._motor

        self._sensor, self._motor = sensor, motor
        self._reward = reward

        self._iteration += 1

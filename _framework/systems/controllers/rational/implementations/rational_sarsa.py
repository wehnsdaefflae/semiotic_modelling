# coding=utf-8
import random
from typing import Tuple

from _framework.data_types import RATIONAL_SENSOR, RATIONAL_MOTOR
from _framework.systems.controllers.rational.abstract import RationalController


class RationalSarsa(RationalController):
    def __init__(self, motor_range: Tuple[Tuple[float, float], ...], epsilon: float, alpha: float, gamma: float):
        super().__init__(motor_range)
        self._epsilon = epsilon
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

        self._iteration = 0

    def react(self, sensor: RATIONAL_SENSOR) -> RATIONAL_MOTOR:
        return self._random_action() if random.random() < self._epsilon else self._actor(sensor)

    def _integrate(self, sensor: RATIONAL_SENSOR, motor: RATIONAL_MOTOR, reward: float):
        if self._iteration >= 1:
            this_condition = sensor, motor
            evaluation = self._critic.predict(this_condition)

            update_value = self._reward + self._gamma * evaluation

            last_condition = self._last_sensor, self._last_motor
            self._critic.fit(last_condition, update_value)

            _last_perception = self._critic.get_state(), self._last_sensor

            if self._iteration >= 2:
                best_last_action = self._actor.predict(self._last_sensor)                   # _last_perception
                best_eval = self._critic(best_last_action, self._last_motor)

                if best_eval < update_value:
                    self._actor.fit(self._last_sensor, (self._last_motor, update_value))    # _last_perception

            self._last_sensor, self._last_motor = self._sensor, self._motor

        self._sensor, self._motor = sensor, motor
        self._reward = reward

        self._iteration += 1

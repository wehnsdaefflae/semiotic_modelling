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

        self._perception = None
        self._action = None
        self._reward = 0.
        self._last_condition = None

        # TODO: choose appropriate predictors
        self._evaluation_predictor = None        # approximate S x M -> float    # RationalPredictor
        self._best_action_predictor = None       # approximate S -> M, float     # RationalPredictor (alternative input: state of evaluation predictor + sensor)

        self._iteration = 0

    def react(self, perception: RATIONAL_SENSOR) -> RATIONAL_MOTOR:
        if random.random() < self._epsilon:
            return self._random_action()
        action, _ = self._best_action_predictor(perception)
        return action

    def _integrate(self, perception: RATIONAL_SENSOR, action: RATIONAL_MOTOR, reward: float):
        if self._iteration >= 1:
            this_condition = perception, action
            evaluation = self._evaluation_predictor.predict(this_condition)

            update_value = self._reward + self._gamma * evaluation

            last_condition = self._last_perception, self._last_action
            self._evaluation_predictor.fit(last_condition, update_value)

            if self._iteration >= 2:
                best_action, best_eval = self._best_action_predictor.predict(self._last_perception, self._last_action)

                if best_eval < update_value:
                    self._best_action_predictor.fit(self._last_perception, (self._last_action, update_value))

            self._last_perception, self._last_action = self._perception, self._action

        self._perception, self._action = perception, action
        self._reward = reward

        self._iteration += 1

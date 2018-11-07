# coding=utf-8
import random
from typing import Tuple, Optional

from _framework.systems.controllers.abstract import Controller, MOTOR_TYPE
from _framework.data_types import RATIONAL_SENSOR, RATIONAL_MOTOR


class RationalController(Controller[RATIONAL_SENSOR, RATIONAL_MOTOR]):
    def __init__(self, motor_range: Tuple[Tuple[float, float], ...], exploration_mean: float, exploration_deviation: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._motor_range = motor_range
        self._mean = exploration_mean
        self._deviation = exploration_deviation

    def _decide(self, perception: RATIONAL_SENSOR) -> RATIONAL_MOTOR:
        raise NotImplementedError()

    def decide(self, perception: RATIONAL_SENSOR) -> RATIONAL_MOTOR:
        target = self._decide(perception)
        return tuple(_t + random.normalvariate(self._mean, self._deviation) for _t in target)

    def _integrate(self, perception: RATIONAL_SENSOR, action: RATIONAL_MOTOR, reward: float):
        raise NotImplementedError()

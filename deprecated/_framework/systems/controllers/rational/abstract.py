# coding=utf-8
import random
from typing import Tuple, Optional

from _framework.systems.controllers.abstract import Controller, MOTOR_TYPE
from _framework.data_types import RATIONAL_SENSOR, RATIONAL_MOTOR
from tools.functionality import clip


class RationalController(Controller[RATIONAL_SENSOR, RATIONAL_MOTOR]):
    def __init__(self, motor_range: Tuple[Tuple[float, float], ...], epsilon: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._motor_range = motor_range
        self._epsilon = epsilon

    def _decide(self, perception: RATIONAL_SENSOR) -> RATIONAL_MOTOR:
        raise NotImplementedError()

    def decide(self, perception: RATIONAL_SENSOR) -> RATIONAL_MOTOR:
        target = self._decide(perception)
        return tuple(
            clip(_t + random.normalvariate(0., self._epsilon), *_ranges)
            for _t, _ranges in zip(target, self._motor_range)
        )

    def _integrate(self, perception: RATIONAL_SENSOR, action: RATIONAL_MOTOR, reward: float):
        raise NotImplementedError()

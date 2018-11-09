# coding=utf-8
import random
from typing import Tuple, Optional

from _framework.systems.controllers.abstract import Controller, MOTOR_TYPE
from _framework.data_types import RATIONAL_SENSOR, RATIONAL_MOTOR


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
            min(max(_t + random.normalvariate(0., self._epsilon), _r[0]), _r[1])
            for _r, _t in zip(self._motor_range, target)
        )

    def _integrate(self, perception: RATIONAL_SENSOR, action: RATIONAL_MOTOR, reward: float):
        raise NotImplementedError()

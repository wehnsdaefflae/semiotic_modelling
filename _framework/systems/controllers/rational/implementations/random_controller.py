# coding=utf-8
import random
from typing import Tuple

from _framework.data_types import RATIONAL_SENSOR, RATIONAL_MOTOR
from _framework.systems.controllers.rational.abstract import RationalController


class RandomRational(RationalController):
    def __init__(self, motor_range: Tuple[Tuple[float, float], ...], *args, **kwargs):
        super().__init__(motor_range, 0., *args, **kwargs)

    def _decide(self, perception: RATIONAL_SENSOR) -> RATIONAL_MOTOR:
        return tuple(
            random.uniform(*_range)
            for _range in self._motor_range
        )

    def _integrate(self, perception: RATIONAL_SENSOR, action: RATIONAL_MOTOR, reward: float):
        pass

# coding=utf-8
import random
from typing import Tuple, Optional

from _framework.systems.controllers.abstract import Controller, MOTOR_TYPE
from _framework.data_types import RATIONAL_SENSOR, RATIONAL_MOTOR


class NominalController(Controller[RATIONAL_SENSOR, RATIONAL_MOTOR]):
    def __init__(self, motor_range: Tuple[Tuple[float, float], ...], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._motor_range = motor_range

    def _random_action(self) -> MOTOR_TYPE:
        return tuple(random.uniform(_min, _max) for _min, _max in self._motor_range)

    def react(self, perception: Optional[RATIONAL_SENSOR]) -> RATIONAL_MOTOR:
        raise NotImplementedError()

    def integrate(self, last_perception: Optional[RATIONAL_SENSOR], last_action: RATIONAL_MOTOR, perception: Optional[RATIONAL_SENSOR], action: RATIONAL_MOTOR, reward: float):
        raise NotImplementedError()

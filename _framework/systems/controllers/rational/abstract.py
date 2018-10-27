# coding=utf-8
from typing import Tuple, Optional

from _framework.systems.controllers.abstract import Controller
from _framework.data_types import RATIONAL_SENSOR, RATIONAL_MOTOR


class NominalController(Controller[RATIONAL_SENSOR, RATIONAL_MOTOR]):
    def __init__(self, motor_range: Tuple[Tuple[float, float], ...], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._motor_range = motor_range

    def _react(self, data_in: RATIONAL_SENSOR) -> RATIONAL_MOTOR:
        raise NotImplementedError()

    def integrate(self, data_in: Optional[RATIONAL_SENSOR], evaluation: float):
        raise NotImplementedError()

    def get_state(self) -> Tuple[Tuple[int, ...], ...]:
        raise NotImplementedError()

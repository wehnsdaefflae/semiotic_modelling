# coding=utf-8
from typing import Tuple, Optional, Collection

from _framework.systems.controllers.abstract import Controller
from _framework.data_types import NOMINAL_MOTOR, NOMINAL_SENSOR


class NominalController(Controller[NOMINAL_SENSOR, NOMINAL_MOTOR]):
    def __init__(self, motor_space: Collection[NOMINAL_MOTOR], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._motor_space = motor_space

    def _react(self, data_in: NOMINAL_SENSOR) -> NOMINAL_MOTOR:
        raise NotImplementedError()

    def integrate(self, data_in: Optional[NOMINAL_SENSOR], reward: float):
        raise NotImplementedError()

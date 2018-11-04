# coding=utf-8
import random
from typing import Collection, Optional

from _framework.data_types import NOMINAL_MOTOR, NOMINAL_SENSOR
from _framework.systems.controllers.nominal.abstract import NominalController


class NominalRandomController(NominalController):
    def __init__(self, motor_space: Collection[NOMINAL_MOTOR]):
        super().__init__(motor_space)

    def _integrate(self, perception: NOMINAL_SENSOR, action: NOMINAL_MOTOR, reward: float):
        pass

    def react(self, _perception: NOMINAL_SENSOR) -> NOMINAL_MOTOR:
        action, = random.sample(self._motor_space, 1)
        return action

# coding=utf-8
import random
from typing import Collection, Optional

from _framework.systems.controllers.abstract import Controller, MOTOR_TYPE
from _framework.data_types import NOMINAL_MOTOR, NOMINAL_SENSOR


class NominalController(Controller[NOMINAL_SENSOR, NOMINAL_MOTOR]):
    def __init__(self, motor_space: Collection[NOMINAL_MOTOR], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._motor_space = motor_space

    def _random_action(self) -> MOTOR_TYPE:
        action, = random.sample(self._motor_space, 1)
        return action

    def react(self, perception: NOMINAL_SENSOR) -> NOMINAL_MOTOR:
        raise NotImplementedError()

    def _integrate(self, perception: NOMINAL_SENSOR, action: NOMINAL_MOTOR, reward: float):
        raise NotImplementedError()
    
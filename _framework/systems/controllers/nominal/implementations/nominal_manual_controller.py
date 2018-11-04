# coding=utf-8
from typing import Collection, Optional

from _framework.data_types import NOMINAL_MOTOR, NOMINAL_SENSOR
from _framework.systems.controllers.nominal.abstract import NominalController
from tools.logger import Logger


class NominalManualController(NominalController):
    def __init__(self, motor_space: Collection[NOMINAL_MOTOR], *args, **kwargs):
        super().__init__(motor_space, *args, **kwargs)
        self._space_string = str(list(sorted(self._motor_space)))

    def react(self, perception: NOMINAL_SENSOR) -> NOMINAL_MOTOR:
        Logger.log(f"\nController {id(self):d} perceives:\n{str(perception):s}")
        action = input(f"Target action {self._space_string:s}: ")
        while action not in self._motor_space:
            action = input(f"Action {action:s} is not among {self._space_string}. Try again: ")
        return action

    def _integrate(self, perception: NOMINAL_SENSOR, action: NOMINAL_MOTOR, reward: float):
        Logger.log(f"### Controller {id(self):d} received reward: {reward:f}.")

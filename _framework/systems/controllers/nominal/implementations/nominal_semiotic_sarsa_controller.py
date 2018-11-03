# coding=utf-8
from typing import Collection, Optional

from _framework.data_types import NOMINAL_MOTOR, NOMINAL_SENSOR
from _framework.systems.controllers.nominal.abstract import NominalController


class NominalSemioticSarsaController(NominalController):
    def __init__(self, motor_space: Collection[NOMINAL_MOTOR], *args, **kwargs):
        super().__init__(motor_space, *args, **kwargs)
        raise NotImplementedError()

    def react(self, perception: Optional[NOMINAL_SENSOR]) -> NOMINAL_MOTOR:
        raise NotImplementedError()

    def integrate(self, perception: Optional[NOMINAL_SENSOR], action: NOMINAL_MOTOR, reward: float):
        raise NotImplementedError()
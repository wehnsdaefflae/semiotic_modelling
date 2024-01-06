# coding=utf-8
from typing import Collection

from _framework.data_types import NOMINAL_MOTOR, NOMINAL_SENSOR
from _framework.systems.controllers.nominal.abstract import NominalController


class NominalSemioticSarsaController(NominalController):
    def __init__(self, motor_space: Collection[NOMINAL_MOTOR], *args, **kwargs):
        super().__init__(motor_space, *args, **kwargs)
        raise NotImplementedError()

    def decide(self, perception: NOMINAL_SENSOR) -> NOMINAL_MOTOR:
        raise NotImplementedError()

    def _integrate(self, perception: NOMINAL_SENSOR, action: NOMINAL_MOTOR, reward: float):
        raise NotImplementedError()

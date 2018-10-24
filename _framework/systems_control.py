# coding=utf-8
import random
from typing import Any, Type, Collection, Hashable, Tuple

from _framework.systems_abstract import Controller, Predictor

NOMINAL_MOTOR = Hashable
NOMINAL_SENSOR = Hashable

RATIONAL_MOTOR = Tuple[float, ...]
RATIONAL_SENSOR = Tuple[float, ...]


class NominalNoneController(Controller[NOMINAL_SENSOR, Type[None]]):
    def __init__(self):
        super().__init__()

    def _integrate(self, evaluation: float):
        return

    def _react(self, data_in: Any) -> Type[None]:
        return None


class NominalRandomController(Controller[NOMINAL_SENSOR, NOMINAL_MOTOR]):
    def __init__(self, motor_space: Collection[NOMINAL_MOTOR]):
        self.motor_space = motor_space

    def _integrate(self, evaluation: float):
        return

    def _react(self, data_in: NOMINAL_SENSOR) -> NOMINAL_MOTOR:
        action, = random.sample(self.motor_space, 1)
        return action


class NominalSarsaController(Controller[NOMINAL_SENSOR, NOMINAL_MOTOR]):
    def __init__(self):
        raise NotImplementedError()

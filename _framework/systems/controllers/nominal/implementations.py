# coding=utf-8
import random
from typing import Any, Type, Collection, Tuple, Optional

from _framework.systems.controllers.nominal.abstract import NominalController
from _framework.data_types import NOMINAL_SENSOR, NOMINAL_MOTOR


class NominalNoneController(NominalController):
    def __init__(self):
        super().__init__()

    def integrate(self, data_in: Optional[NOMINAL_SENSOR], evaluation: float):
        pass

    def _react(self, data_in: Any) -> Type[None]:
        return None

    def get_state(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple()


class NominalRandomController(NominalController):
    def __init__(self, motor_space: Collection[NOMINAL_MOTOR]):
        super().__init__()
        self.motor_space = motor_space

    def integrate(self, data_in: Optional[NOMINAL_SENSOR], evaluation: float):
        pass

    def _react(self, data_in: NOMINAL_SENSOR) -> NOMINAL_MOTOR:
        action, = random.sample(self.motor_space, 1)
        return action

    def get_state(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple()


class NominalSarsaController(NominalController):
    def __init__(self):
        super().__init__()
        raise NotImplementedError()

    def get_state(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple()

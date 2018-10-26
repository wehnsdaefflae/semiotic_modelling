# coding=utf-8
from typing import Hashable, Tuple, Collection

from _framework.systems_abstract import Task

NOMINAL_SENSOR = Hashable
NOMINAL_MOTOR = Hashable

RATIONAL_MOTOR = Tuple[float, ...]
RATIONAL_SENSOR = Tuple[float, ...]


class NominalGridWorld(Task[NOMINAL_MOTOR, NOMINAL_SENSOR]):
    def __init__(self):
        super().__init__()
        raise NotImplementedError()

    def _react(self, data_in: NOMINAL_MOTOR) -> NOMINAL_SENSOR:
        raise NotImplementedError()

    def _evaluate_action(self, data_in: NOMINAL_MOTOR) -> float:
        raise NotImplementedError()

    def get_state(self) -> Tuple[Tuple[int, ...], ...]:
        raise NotImplementedError()

    @staticmethod
    def motor_space() -> Collection[NOMINAL_MOTOR]:
        return {True, False}

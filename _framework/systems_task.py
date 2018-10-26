# coding=utf-8
from typing import Hashable, Tuple, Collection

from _framework.systems_abstract import Task

NOMINAL_SENSOR = Hashable
NOMINAL_MOTOR = Hashable

RATIONAL_MOTOR = Tuple[float, ...]
RATIONAL_SENSOR = Tuple[float, ...]


class NominalMyTask(Task[NOMINAL_MOTOR, NOMINAL_SENSOR]):
    def __init__(self):
        pass

    def _react(self, data_in: NOMINAL_MOTOR) -> NOMINAL_SENSOR:
        pass

    def _evaluate_action(self, data_in: NOMINAL_MOTOR) -> float:
        pass

    def get_state(self) -> Tuple[Tuple[int, ...], ...]:
        pass

    @staticmethod
    def motor_space() -> Collection[NOMINAL_MOTOR]:
        return {True, False}

# coding=utf-8
from typing import Tuple, Collection

from _framework.data_types import NOMINAL_MOTOR, NOMINAL_SENSOR
from _framework.systems.tasks.nominal.abstract import NominalTask


class NominalGridWorld(NominalTask):
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

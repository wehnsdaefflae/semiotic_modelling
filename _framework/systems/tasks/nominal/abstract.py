# coding=utf-8
from typing import Collection, Optional

from _framework.data_types import NOMINAL_MOTOR, NOMINAL_SENSOR
from _framework.systems.tasks.abstract import Task


class NominalTask(Task[NOMINAL_MOTOR, NOMINAL_SENSOR]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def react(self, data_in: Optional[NOMINAL_MOTOR]) -> NOMINAL_SENSOR:
        raise NotImplementedError()

    def _evaluate_action(self, data_in: Optional[NOMINAL_MOTOR]) -> float:
        raise NotImplementedError()

    @staticmethod
    def motor_space() -> Collection[NOMINAL_MOTOR]:
        raise NotImplementedError()

# coding=utf-8
from typing import Tuple, Optional

from _framework.data_types import RATIONAL_MOTOR, RATIONAL_SENSOR
from _framework.systems.tasks.abstract import Task


class RationalTask(Task[RATIONAL_MOTOR, RATIONAL_SENSOR]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def react(self, data_in: Optional[RATIONAL_MOTOR]) -> RATIONAL_SENSOR:
        raise NotImplementedError()

    @staticmethod
    def motor_range() -> Tuple[Tuple[float, float], ...]:
        raise NotImplementedError()

    def _get_reward(self) -> float:
        raise NotImplementedError()

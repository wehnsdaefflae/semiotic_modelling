# coding=utf-8
from typing import TypeVar, Optional, Tuple, Generic

from _framework.systems.abstract import System

SENSOR_TYPE = TypeVar("SENSOR_TYPE")
MOTOR_TYPE = TypeVar("MOTOR_TYPE")


class Task(System[MOTOR_TYPE, SENSOR_TYPE], Generic[MOTOR_TYPE, SENSOR_TYPE]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def react(self, data_in: Optional[MOTOR_TYPE]) -> SENSOR_TYPE:
        raise NotImplementedError()

    def _get_reward(self) -> float:
        raise NotImplementedError()

    def respond(self, data_in: Optional[MOTOR_TYPE]) -> Tuple[SENSOR_TYPE, float]:
        return self.react(data_in), self._get_reward()

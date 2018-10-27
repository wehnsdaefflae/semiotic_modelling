# coding=utf-8
from typing import TypeVar, Optional, Tuple, Generic

from _framework.systems.abstract import System

SENSOR_TYPE = TypeVar("SENSOR_TYPE")
MOTOR_TYPE = TypeVar("MOTOR_TYPE")


class Task(System[MOTOR_TYPE, SENSOR_TYPE], Generic[MOTOR_TYPE, SENSOR_TYPE]):
    def __init__(self, *args, **kwargs):
        pass

    def _react(self, data_in: MOTOR_TYPE) -> SENSOR_TYPE:
        raise NotImplementedError()

    def _evaluate_action(self, data_in: MOTOR_TYPE) -> float:
        raise NotImplementedError()

    def respond(self, data_in: Optional[MOTOR_TYPE]) -> Tuple[SENSOR_TYPE, float]:
        return self._react(data_in), self._evaluate_action(data_in)

    def get_state(self) -> Tuple[Tuple[int, ...], ...]:
        raise NotImplementedError()

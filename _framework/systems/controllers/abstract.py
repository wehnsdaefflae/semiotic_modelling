# coding=utf-8
from typing import Optional, Tuple, TypeVar, Generic

from _framework.systems.abstract import System


SENSOR_TYPE = TypeVar("SENSOR_TYPE")
MOTOR_TYPE = TypeVar("MOTOR_TYPE")


class Controller(System[SENSOR_TYPE, MOTOR_TYPE], Generic[SENSOR_TYPE, MOTOR_TYPE]):
    def __init__(self, *args, **kwargs):
        pass

    def _react(self, data_in: SENSOR_TYPE) -> MOTOR_TYPE:
        raise NotImplementedError()

    def integrate(self, data_in: Optional[SENSOR_TYPE], evaluation: float):
        raise NotImplementedError()

    def decide(self, data_in: Optional[SENSOR_TYPE]) -> MOTOR_TYPE:
        return self._react(data_in)

    def get_state(self) -> Tuple[Tuple[int, ...], ...]:
        raise NotImplementedError()
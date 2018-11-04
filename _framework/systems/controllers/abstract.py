# coding=utf-8
from typing import Optional, Tuple, TypeVar, Generic

from _framework.systems.abstract import System


SENSOR_TYPE = TypeVar("SENSOR_TYPE")
MOTOR_TYPE = TypeVar("MOTOR_TYPE")


class Controller(System[SENSOR_TYPE, MOTOR_TYPE], Generic[SENSOR_TYPE, MOTOR_TYPE]):
    def __init__(self, *args, **kwargs):
        pass

    def _random_action(self) -> MOTOR_TYPE:
        raise NotImplementedError()

    def react(self, perception: SENSOR_TYPE) -> MOTOR_TYPE:
        raise NotImplementedError()

    def _integrate(self, perception: SENSOR_TYPE, action: MOTOR_TYPE, reward: float):
        raise NotImplementedError()

    def integrate(self, perception: Optional[SENSOR_TYPE], action: MOTOR_TYPE, reward: float):
        if perception is None:
            return
        self._integrate(perception, action, reward)

    def decide(self, perception: Optional[SENSOR_TYPE]) -> MOTOR_TYPE:
        if perception is None:
            action = self._random_action()
        else:
            action = self.react(perception)

        return action

# coding=utf-8
from typing import Optional, TypeVar, Generic

from _framework.systems.abstract import System
from tools.functionality import smear

SENSOR_TYPE = TypeVar("SENSOR_TYPE")
MOTOR_TYPE = TypeVar("MOTOR_TYPE")


class Controller(System[SENSOR_TYPE, MOTOR_TYPE], Generic[SENSOR_TYPE, MOTOR_TYPE]):
    def __init__(self, *args, **kwargs):
        self.__iteration = 0
        self.average_reward = 0.

    def get_iterations(self) -> int:
        return self.__iteration

    def react(self, perception: SENSOR_TYPE) -> MOTOR_TYPE:
        motor = self.decide(perception)
        self.__iteration += 1
        return motor

    def _integrate(self, perception: SENSOR_TYPE, action: MOTOR_TYPE, reward: float):
        raise NotImplementedError()

    def integrate(self, perception: Optional[SENSOR_TYPE], action: MOTOR_TYPE, reward: float):
        self.average_reward = smear(self.average_reward, reward, self.__iteration)
        if perception is None:
            return
        self._integrate(perception, action, reward)

    def decide(self, perception: Optional[SENSOR_TYPE]) -> MOTOR_TYPE:
        raise NotImplementedError()

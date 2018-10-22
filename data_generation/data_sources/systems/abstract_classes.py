#!/usr/bin/env python3
# coding=utf-8
from typing import TypeVar, Generic, Tuple, Optional


MOTOR_TYPE = TypeVar("MOTOR_TYPE")
SENSOR_TYPE = TypeVar("SENSOR_TYPE")


class System(Generic[MOTOR_TYPE, SENSOR_TYPE]):
    def __init__(self, *args, **kwargs):
        pass

    def get_motor_range(self) -> Generic[MOTOR_TYPE]:
        raise NotImplementedError()

    def react_to(self, motor: Optional[MOTOR_TYPE]) -> Tuple[SENSOR_TYPE, float]:
        raise NotImplementedError()


INTERACTION_CONDITION = Tuple[SENSOR_TYPE, MOTOR_TYPE]
INTERACTION_HISTORY = Tuple[INTERACTION_CONDITION, ...]


class Controller(Generic[SENSOR_TYPE, MOTOR_TYPE]):
    def __init__(self, motor_range: Generic[MOTOR_TYPE]):
        self.motor_range = motor_range

    def react_to(self, sensor: SENSOR_TYPE, reward: float) -> MOTOR_TYPE:
        raise NotImplementedError()


EXAMPLE_INPUT = TypeVar("EXAMPLE_INPUT")
EXAMPLE_OUTPUT = TypeVar("EXAMPLE_OUTPUT")


class DeprecatedExampleFactory(Generic[EXAMPLE_INPUT, EXAMPLE_OUTPUT]):
    def get_example(self) -> Tuple[EXAMPLE_INPUT, EXAMPLE_OUTPUT]:
        raise NotImplementedError()

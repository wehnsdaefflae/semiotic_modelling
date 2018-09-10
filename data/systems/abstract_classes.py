#!/usr/bin/env python3
# coding=utf-8
from typing import TypeVar, Generic, Optional, Tuple

INPUT_DATA = TypeVar("INPUT_DATA")
OUTPUT_DATA = TypeVar("OUTPUT_DATA")


class System(Generic[INPUT_DATA, OUTPUT_DATA]):
    def react_to(self, input_data: INPUT_DATA) -> OUTPUT_DATA:
        raise NotImplementedError()


MOTOR_TYPE = TypeVar("MOTOR_TYPE")
SENSOR_TYPE = TypeVar("SENSOR_TYPE")
#EXPERIENCE = Tuple[SENSOR_TYPE, float]


class EXPERIENCE(Tuple[SENSOR_TYPE, float]):
    ...


class Environment(System[MOTOR_TYPE, SENSOR_TYPE]):
    def react_to(self, motor: Optional[MOTOR_TYPE]) -> EXPERIENCE[SENSOR_TYPE]:
        raise NotImplementedError()


class Controller(System[SENSOR_TYPE, MOTOR_TYPE]):
    def __init__(self, motor_range: Generic[MOTOR_TYPE]):
        self.motor_range = motor_range

    def react_to(self, experience: EXPERIENCE[SENSOR_TYPE]) -> MOTOR_TYPE:
        raise NotImplementedError()

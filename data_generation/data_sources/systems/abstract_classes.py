#!/usr/bin/env python3
# coding=utf-8
from typing import TypeVar, Generic, Tuple, Optional

from modelling.predictors.abstract_predictor import Predictor


INPUT_DATA = TypeVar("INPUT_DATA")
OUTPUT_DATA = TypeVar("OUTPUT_DATA")


class System(Generic[INPUT_DATA, OUTPUT_DATA]):
    def react_to(self, input_data: INPUT_DATA) -> OUTPUT_DATA:
        raise NotImplementedError()


MOTOR_TYPE = TypeVar("MOTOR_TYPE")
SENSOR_TYPE = TypeVar("SENSOR_TYPE")


# EXPERIENCE = Tuple[SENSOR_TYPE, float]


class EXPERIENCE(Tuple[SENSOR_TYPE, float]):
    ...


MOTOR_MAYBE = Optional[MOTOR_TYPE]


class Environment(System[MOTOR_MAYBE, EXPERIENCE]):
    def react_to(self, motor: MOTOR_MAYBE) -> EXPERIENCE:
        raise NotImplementedError()


INTERACTION_CONDITION = Tuple[SENSOR_TYPE, MOTOR_TYPE]
INTERACTION_HISTORY = Tuple[INTERACTION_CONDITION, ...]


class Controller(System[EXPERIENCE, MOTOR_TYPE]):
    def __init__(self, motor_range: Generic[MOTOR_TYPE]):
        self.motor_range = motor_range

    def react_to(self, experience: EXPERIENCE) -> MOTOR_TYPE:
        raise NotImplementedError()


EXAMPLE_INPUT = TypeVar("EXAMPLE_INPUT")
EXAMPLE_OUTPUT = TypeVar("EXAMPLE_OUTPUT")


class DeprecatedExampleFactory(Generic[EXAMPLE_INPUT, EXAMPLE_OUTPUT]):
    def get_example(self) -> Tuple[EXAMPLE_INPUT, EXAMPLE_OUTPUT]:
        raise NotImplementedError()

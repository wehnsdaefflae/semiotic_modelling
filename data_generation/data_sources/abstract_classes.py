#!/usr/bin/env python3
# coding=utf-8
from typing import TypeVar, Generic, Tuple, Optional

from modelling.predictors.abstract_predictor import Predictor

ELEMENT = TypeVar("ELEMENT")


class Sequence(Generic[ELEMENT]):
    def __init__(self, history_length: int = 1):
        self.history_length = history_length
        self.history = []

    def get_example(self) -> Tuple[Tuple[ELEMENT, ...], ELEMENT]:
        for _ in range(self.history_length - len(self.history)):
            self.history.append(self._next_element())

        fixed_history = tuple(self.history)
        element = self._next_element()

        self.history.append(element)
        for _ in range(len(self.history) - self.history_length + 1):
            self.history.pop(0)

        return fixed_history, element

    def _next_element(self) -> ELEMENT:
        raise NotImplementedError()


INPUT_DATA = TypeVar("INPUT_DATA")
OUTPUT_DATA = TypeVar("OUTPUT_DATA")


class System(Generic[INPUT_DATA, OUTPUT_DATA]):
    def react_to(self, input_data: Optional[INPUT_DATA]) -> OUTPUT_DATA:
        raise NotImplementedError()


MOTOR_TYPE = TypeVar("MOTOR_TYPE")
SENSOR_TYPE = TypeVar("SENSOR_TYPE")


# EXPERIENCE = Tuple[SENSOR_TYPE, float]


class EXPERIENCE(Tuple[SENSOR_TYPE, float]):
    ...


class Environment(System[MOTOR_TYPE, EXPERIENCE]):
    def react_to(self, motor: MOTOR_TYPE) -> EXPERIENCE:
        raise NotImplementedError()


INTERACTION_CONDITION = Tuple[SENSOR_TYPE, MOTOR_TYPE]
INTERACTION_HISTORY = Tuple[INTERACTION_CONDITION, ...]


class Controller(System[EXPERIENCE, MOTOR_TYPE]):
    def __init__(self, motor_range: Generic[MOTOR_TYPE], predictor: Predictor[INTERACTION_HISTORY, SENSOR_TYPE] = None):
        self.motor_range = motor_range
        self.predictor = predictor

    def react_to(self, experience: EXPERIENCE) -> MOTOR_TYPE:
        raise NotImplementedError()


EXAMPLE_INPUT = TypeVar("EXAMPLE_INPUT")
EXAMPLE_OUTPUT = TypeVar("EXAMPLE_OUTPUT")


class ExampleFactory(Generic[EXAMPLE_INPUT, EXAMPLE_OUTPUT]):
    def get_example(self) -> Tuple[EXAMPLE_INPUT, EXAMPLE_OUTPUT]:
        raise NotImplementedError()
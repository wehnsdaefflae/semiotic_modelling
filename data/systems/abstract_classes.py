#!/usr/bin/env python3
# coding=utf-8
from typing import TypeVar, Generic, Optional, Tuple

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
    def react_to(self, input_data: INPUT_DATA) -> OUTPUT_DATA:
        raise NotImplementedError()


MOTOR_TYPE = TypeVar("MOTOR_TYPE")
SENSOR_TYPE = TypeVar("SENSOR_TYPE")


# EXPERIENCE = Tuple[SENSOR_TYPE, float]


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


EXAMPLE_INPUT = TypeVar("EXAMPLE_INPUT")
EXAMPLE_OUTPUT = TypeVar("EXAMPLE_OUTPUT")


class ExampleFactory(Generic[EXAMPLE_INPUT, EXAMPLE_OUTPUT]):
    def get_example(self) -> Tuple[EXAMPLE_INPUT, EXAMPLE_OUTPUT]:
        raise NotImplementedError()

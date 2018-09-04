#!/usr/bin/env python3
# coding=utf-8
import random
from typing import Iterable, Generator, Tuple, TypeVar, List, Optional
from data.data_types import CONCURRENT_EXAMPLES, EXAMPLE

EXAMPLE_SEQUENCE = Iterable[EXAMPLE]
HOMOGENEOUS_TYPE = TypeVar("HOMOGENEOUS_TYPE")


def example_sequence(source: Iterable[HOMOGENEOUS_TYPE], history_length: int) -> EXAMPLE_SEQUENCE[HOMOGENEOUS_TYPE, HOMOGENEOUS_TYPE]:
    history: List[HOMOGENEOUS_TYPE] = []
    for each_value in source:
        if len(history) == history_length:
            input_value = tuple(history)
            target_value = each_value,
            example = input_value, target_value   # type: EXAMPLE[HOMOGENEOUS_TYPE, HOMOGENEOUS_TYPE] # TODO: what's going on here?
            yield example

        history.append(each_value)
        while history_length < len(history):
            history.pop(0)


SENSOR_TYPE = TypeVar("SENSOR_TYPE")
MOTOR_TYPE = TypeVar("MOTOR_TYPE")
INTERACTION_HISTORY = Tuple[Tuple[SENSOR_TYPE, MOTOR_TYPE], ...]

ENVIRONMENT = Generator[SENSOR_TYPE, Optional[MOTOR_TYPE], None]

GRID_SENSOR_VALUE = Tuple[str, str, str, str]
GRID_MOTOR_VALUE = str
GRID_HISTORY = INTERACTION_HISTORY[GRID_SENSOR_VALUE, GRID_MOTOR_VALUE]


def example_random_interactive(source: ENVIRONMENT[GRID_SENSOR_VALUE, GRID_MOTOR_VALUE],
                               actions: Tuple[GRID_MOTOR_VALUE, ...],
                               history_length: int) -> CONCURRENT_EXAMPLES[GRID_HISTORY, GRID_SENSOR_VALUE]:
    history = []    # type: List[Tuple[GRID_SENSOR_VALUE, GRID_MOTOR_VALUE]]

    each_sensor = source.send(None)
    while True:
        each_motor = random.choice(actions)

        each_condition = each_sensor, each_motor
        history.append(each_condition)
        while history_length < len(history):
            history.pop(0)

        each_sensor = source.send(each_motor)
        if len(history) == history_length:
            yield (tuple(history), each_sensor),


GRID_SENSOR_VALUE = str
GRID_HISTORY = INTERACTION_HISTORY[GRID_SENSOR_VALUE, GRID_MOTOR_VALUE]


def example_random_interactive_senses(source: ENVIRONMENT[GRID_SENSOR_VALUE, GRID_MOTOR_VALUE],
                                      actions: Tuple[GRID_MOTOR_VALUE, ...],
                                      history_length: int) -> CONCURRENT_EXAMPLES[GRID_HISTORY, GRID_SENSOR_VALUE]:
    histories = [], [], [], []

    each_sensor = source.send(None)
    while True:
        each_motor = random.choice(actions)

        for sensor_index, each_history in enumerate(histories):
            each_condition = each_sensor[sensor_index], each_motor
            each_history.append(each_condition)
            while history_length < len(each_history):
                each_history.pop(0)

        each_sensor = source.send(each_motor)
        if all(len(each_history) == history_length for each_history in histories):
            yield tuple((tuple(each_history), each_sensor[sensor_index]) for sensor_index, each_history in enumerate(histories))


def example_goal_interactive(source: Generator[Tuple[SENSOR_TYPE, ...], Optional[MOTOR_TYPE], None],
                             actions: Tuple[MOTOR_TYPE, ...],
                             history_length: int) -> \
        Generator[Tuple[EXAMPLE[Tuple[Tuple[SENSOR_TYPE, MOTOR_TYPE], ...], SENSOR_TYPE], float], None, None]:
    raise NotImplementedError("also returns current reward")


if __name__ == "__main__":
    pass

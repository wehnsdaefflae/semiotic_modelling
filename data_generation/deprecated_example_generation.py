#!/usr/bin/env python3
# coding=utf-8
from typing import Iterable, Generator, Tuple, TypeVar, List, Optional, Hashable, Dict


from tools.math_functions import distribute_circular, flatten

INPUT_TYPE = TypeVar("INPUT_TYPE")
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")
HOMOGENEOUS_TYPE = TypeVar("HOMOGENEOUS_TYPE")

EXAMPLE = Tuple[INPUT_TYPE, OUTPUT_TYPE]
CONCURRENT_EXAMPLES = Iterable[Tuple[EXAMPLE, ...]]

EXAMPLE_SEQUENCE = Iterable[EXAMPLE]


def example_sequence(source: Iterable[HOMOGENEOUS_TYPE], history_length: int) -> EXAMPLE_SEQUENCE[HOMOGENEOUS_TYPE, HOMOGENEOUS_TYPE]:
    history: List[HOMOGENEOUS_TYPE] = []
    for each_value in source:
        if len(history) == history_length:
            input_value = tuple(history)
            target_value = each_value,
            example = input_value, target_value   # type: EXAMPLE[HOMOGENEOUS_TYPE, HOMOGENEOUS_TYPE]
            yield example

        history.append(each_value)
        while history_length < len(history):
            history.pop(0)


SENSOR_TYPE = TypeVar("SENSOR_TYPE")
MOTOR_TYPE = TypeVar("MOTOR_TYPE")
REWARD = float

INTERACTION_HISTORY = Tuple[Tuple[SENSOR_TYPE, MOTOR_TYPE], ...]

GRID_SENSOR_TYPE = Tuple[str, str, str, str]
GRID_MOTOR_TYPE = str
GRID_HISTORY = INTERACTION_HISTORY[GRID_SENSOR_TYPE, GRID_MOTOR_TYPE]

FEEDBACK = Tuple[SENSOR_TYPE, REWARD]
ENVIRONMENT = Generator[FEEDBACK, Optional[MOTOR_TYPE], None]
CONTROLLER = Generator[MOTOR_TYPE, Optional[FEEDBACK], None]


def example_interactive(source: ENVIRONMENT[GRID_SENSOR_TYPE, GRID_MOTOR_TYPE],
                        controller: CONTROLLER[GRID_MOTOR_TYPE, GRID_SENSOR_TYPE],
                        history_length: int) -> CONCURRENT_EXAMPLES[GRID_HISTORY, GRID_SENSOR_TYPE]:
    history = []    # type: List[Tuple[GRID_SENSOR_TYPE, GRID_MOTOR_TYPE]]

    each_motor = controller.send(None)              # type: GRID_MOTOR_TYPE
    each_sensor, each_reward = source.send(None)    # type: FEEDBACK[GRID_SENSOR_TYPE]
    while True:
        each_condition = each_sensor, each_motor
        history.append(each_condition)
        while history_length < len(history):
            history.pop(0)

        feedback = source.send(each_motor)
        each_sensor, _ = feedback
        if len(history) == history_length:
            yield (tuple(history), each_sensor),

        each_motor = controller.send(feedback)


GRID_SENSOR_TYPE = str
GRID_HISTORY = INTERACTION_HISTORY[GRID_SENSOR_TYPE, GRID_MOTOR_TYPE]


def example_interactive_senses(source: ENVIRONMENT[GRID_SENSOR_TYPE, GRID_MOTOR_TYPE],
                               controller: CONTROLLER[GRID_MOTOR_TYPE, GRID_SENSOR_TYPE],
                               history_length: int) -> CONCURRENT_EXAMPLES[GRID_HISTORY, GRID_SENSOR_TYPE]:
    histories = [], [], [], []

    each_motor = controller.send(None)
    each_sensor, each_reward = source.send(None)
    while True:
        for sensor_index, each_history in enumerate(histories):
            each_condition = each_sensor[sensor_index], each_motor
            each_history.append(each_condition)
            while history_length < len(each_history):
                each_history.pop(0)

        feedback = source.send(each_motor)
        each_sensor, _ = feedback
        if all(len(each_history) == history_length for each_history in histories):
            yield tuple((tuple(each_history), each_sensor[sensor_index]) for sensor_index, each_history in enumerate(histories))

        each_motor = controller.send(feedback)


IN_TYPE = Hashable
OUT_TYPE = Hashable

IN_VECTOR = Tuple[float, ...]
OUT_VECTOR = Tuple[float, ...]


def rationalize_generator(source: CONCURRENT_EXAMPLES[IN_TYPE, OUT_TYPE]) -> CONCURRENT_EXAMPLES[IN_VECTOR, OUT_VECTOR]:
    in_values = dict()
    out_values = dict()

    def _convert(value: Hashable, c_dict: Dict[Hashable, float]) -> float:
        r_value = c_dict.get(value)
        if r_value is None:
            r_value = distribute_circular(len(c_dict))
            c_dict[value] = r_value
        return r_value

    for concurrent_examples in source:
        rational_examples = []
        for input_values, output_values in concurrent_examples:
            each_rational_input = tuple(_convert(_x, in_values) for _x in flatten(input_values))
            each_rational_output = tuple(_convert(_x, out_values) for _x in flatten(output_values))
            each_example = each_rational_input, each_rational_output
            rational_examples.append(each_example)
        yield tuple(rational_examples)
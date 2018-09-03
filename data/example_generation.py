#!/usr/bin/env python3
# coding=utf-8
import random
from typing import Iterable, Generator, Tuple, TypeVar, Hashable, List, Any, Optional

INPUT_TYPE = TypeVar("INPUT_TYPE", Hashable, Tuple[float, ...])
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE", Hashable, Tuple[float, ...])

INPUT = Tuple[INPUT_TYPE, ...]
OUTPUT = Tuple[OUTPUT_TYPE, ...]
EXAMPLE = Tuple[INPUT[INPUT_TYPE], OUTPUT[OUTPUT_TYPE]]
EXAMPLE_SEQUENCE = Iterable[EXAMPLE[INPUT_TYPE, OUTPUT_TYPE]]
JOINT_SEQUENCES = Iterable[Tuple[EXAMPLE[INPUT_TYPE, OUTPUT_TYPE], ...]]


def join_sequences(individual_sequences: Tuple[EXAMPLE_SEQUENCE, ...]) -> Generator[Tuple[EXAMPLE, ...], None, None]:
    yield from zip(*individual_sequences)


HOMOGENEOUS_TYPE = TypeVar("HOMOGENEOUS_TYPE")


def example_sequence(source: Iterable[HOMOGENEOUS_TYPE], history_length: int) -> Generator[EXAMPLE[HOMOGENEOUS_TYPE, HOMOGENEOUS_TYPE], None, None]:
    history: List[HOMOGENEOUS_TYPE] = []
    for each_value in source:
        if len(history) == history_length:
            input_value: INPUT = tuple(history)
            target_value: OUTPUT = (each_value,)
            example: EXAMPLE = (input_value, target_value)
            yield example

        history.append(each_value)
        while history_length < len(history):
            history.pop(0)


SENSOR_TYPE = Tuple[str, str, str, str]  # TypeVar("SENSOR_TYPE")
MOTOR_TYPE = str  # TypeVar("MOTOR_TYPE")


def example_random_interactive(source: Generator[SENSOR_TYPE, Optional[MOTOR_TYPE], None], actions: Tuple[MOTOR_TYPE, ...],
                               history_length: int) -> Generator[EXAMPLE[Tuple[Tuple[SENSOR_TYPE, MOTOR_TYPE], ...], SENSOR_TYPE], None, None]:
    history: List[Tuple[SENSOR_TYPE, MOTOR_TYPE]] = []

    each_sensor = source.send(None)

    while True:
        each_motor = random.choice(actions)

        each_condition = each_sensor, each_motor
        history.append(each_condition)
        while history_length < len(history):
            history.pop(0)

        each_sensor = source.send(each_motor)
        if len(history) == history_length:
            yield tuple(history), each_sensor


def example_random_interactive_senses(source: Generator[SENSOR_TYPE, Optional[MOTOR_TYPE], None], actions: Tuple[MOTOR_TYPE, ...],
                                      history_length: int) -> Generator[EXAMPLE[Tuple[Tuple[SENSOR_TYPE, MOTOR_TYPE], ...], SENSOR_TYPE], None, None]:
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


def _join_sequences(individual_sequences: Tuple[Iterable[Any], ...]) -> Generator[Tuple[Any, ...], None, None]:
    for current_examples in zip(*individual_sequences):
        yield current_examples


def _join_sequences_2(individual_sequences: Tuple[Iterable[Any], ...]) -> Generator[Tuple[Any, ...], None, None]:
    yield from zip(*individual_sequences)


def gen_test() -> Generator[int, int, None]:
    _x = 0
    while True:
        _y = yield _x
        _x += 1
        if _y is None:
            continue
        _x += _y


if __name__ == "__main__":
    EXAMPLE_SEQUENCES = \
        Tuple[                          # tuple of parallel sequences
            Iterable[                   # sequence of examples
                Tuple[                  # example of input and target output
                    Tuple[str, ...],    # input of values
                    Tuple[str, ...]     # target output of values
                ]
            ], ...
        ]

    # convert one into another with zip(*one) = another

    EXAMPLES_SEQUENCE = \
        Iterable[                       # sequence of concurrent examples
            Tuple[                      # concurrent examples of individual examples
                Tuple[                  # individual examples of input and target output
                    Tuple[str, ...],    # input of values
                    Tuple[str, ...]     # target output of values
                ], ...
            ]
        ]

    sequences: EXAMPLE_SEQUENCES = \
        ([(("seq1ex1in_val1", "seq1ex1in_val2", "seq1ex1in_val3"), ("seq1ex1out_val1", "seq1ex1out_val2", "seq1ex1out_val3")),
          (("seq1ex2in_val1", "seq1ex2in_val2", "seq1ex2in_val3"), ("seq1ex2out_val1", "seq1ex2out_val2", "seq1ex2out_val3")),
          (("seq1ex3in_val1", "seq1ex3in_val2", "seq1ex3in_val3"), ("seq1ex3out_val1", "seq1ex3out_val2", "seq1ex3out_val3"))],
         [(("seq2ex1in_val1", "seq2ex1in_val2", "seq2ex1in_val3"), ("seq2ex1out_val1", "seq2ex1out_val2", "seq2ex1out_val3")),
          (("seq2ex2in_val1", "seq2ex2in_val2", "seq2ex2in_val3"), ("seq2ex2out_val1", "seq2ex2out_val2", "seq2ex2out_val3")),
          (("seq2ex3in_val1", "seq2ex3in_val2", "seq2ex3in_val3"), ("seq2ex3out_val1", "seq2ex3out_val2", "seq2ex3out_val3"))],
         [(("seq3ex1in_val1", "seq3ex1in_val2", "seq3ex1in_val3"), ("seq3ex1out_val1", "seq3ex1out_val2", "seq3ex1out_val3")),
          (("seq3ex2in_val1", "seq3ex2in_val2", "seq3ex2in_val3"), ("seq3ex2out_val1", "seq3ex2out_val2", "seq3ex2out_val3")),
          (("seq3ex3in_val1", "seq3ex3in_val2", "seq3ex3in_val3"), ("seq3ex3out_val1", "seq3ex3out_val2", "seq3ex3out_val3"))])

    examples: EXAMPLES_SEQUENCE = \
        [((("seq1ex1in_val1", "seq1ex1in_val2", "seq1ex1in_val3"), ("seq1ex1out_val1", "seq1ex1out_val2", "seq1ex1out_val3")),
          (("seq2ex1in_val1", "seq2ex1in_val2", "seq2ex1in_val3"), ("seq2ex1out_val1", "seq2ex1out_val2", "seq2ex1out_val3")),
          (("seq3ex1in_val1", "seq3ex1in_val2", "seq3ex1in_val3"), ("seq3ex1out_val1", "seq3ex1out_val2", "seq3ex1out_val3"))),
         ((("seq1ex2in_val1", "seq1ex2in_val2", "seq1ex2in_val3"), ("seq1ex2out_val1", "seq1ex2out_val2", "seq1ex2out_val3")),
          (("seq2ex2in_val1", "seq2ex2in_val2", "seq2ex2in_val3"), ("seq2ex2out_val1", "seq2ex2out_val2", "seq2ex2out_val3")),
          (("seq3ex2in_val1", "seq3ex2in_val2", "seq3ex2in_val3"), ("seq3ex2out_val1", "seq3ex2out_val2", "seq3ex2out_val3"))),
         ((("seq1ex3in_val1", "seq1ex3in_val2", "seq1ex3in_val3"), ("seq1ex3out_val1", "seq1ex3out_val2", "seq1ex3out_val3")),
          (("seq2ex3in_val1", "seq2ex3in_val2", "seq2ex3in_val3"), ("seq2ex3out_val1", "seq2ex3out_val2", "seq2ex3out_val3")),
          (("seq3ex3in_val1", "seq3ex3in_val2", "seq3ex3in_val3"), ("seq3ex3out_val1", "seq3ex3out_val2", "seq3ex3out_val3")))]

    print(sequences == tuple(list(_x) for _x in zip(*examples)))
    print()
    print(examples == list(zip(*sequences)))

    exit()
    sequences_simple = (["seq1ex1", "seq1ex2", "seq1ex3"], ["seq2ex1", "seq2ex2", "seq2ex3"], ["seq3ex1", "seq3ex2", "seq3ex3"])
    examples_simple = [("seq1ex1", "seq2ex1", "seq3ex1"), ("seq1ex2", "seq2ex2", "seq3ex2"), ("seq1ex3", "seq2ex3", "seq3ex3")]

    print(sequences_simple)
    print(tuple(list(_x) for _x in zip(*examples_simple)))
    print()
    print(examples_simple)
    print(list(zip(*sequences_simple)))


    exit()
    for _x in zip(*sequences):
        for _y in zip(*_x):
            print(_y)
            print()

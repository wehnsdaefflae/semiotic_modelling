# coding=utf-8
from typing import Tuple, Iterator, TypeVar

from data_generation.data_sources.abstract_classes import System

INPUT = TypeVar("INPUT")
TARGET = TypeVar("TARGET")

INPUT_SEQUENCE = Iterator[INPUT]
TARGET_SEQUENCE = Iterator[TARGET]
EXAMPLE_SEQUENCE = Tuple[INPUT_SEQUENCE, TARGET_SEQUENCE]

EXAMPLE = Tuple[INPUT, TARGET]
CONCURRENT_EXAMPLES = Tuple[EXAMPLE, ...]


def from_sequences(sequences: Tuple[EXAMPLE_SEQUENCE, ...]) -> Iterator[CONCURRENT_EXAMPLES]:
    no_examples = len(sequences)

    all_sequence_ids = tuple(id(_seq) for _ex in sequences for _seq in _ex)
    if not len(all_sequence_ids) == len(set(all_sequence_ids)):
        raise ValueError("All sequences must be individual iterator instances.")

    while True:
        examples = tuple((next(_in), next(_tar)) for _in, _tar in sequences)
        if len(examples) != no_examples:
            raise StopIteration()

        yield examples


DATA_IN = TypeVar("DATA_IN")
DATA_OUT = TypeVar("DATA_OUT")

INTERACTION_CONDITION = Tuple[DATA_IN, DATA_OUT]
INTERACTION_HISTORY = Tuple[INTERACTION_CONDITION, ...]


def from_systems(this_system: System[DATA_IN, DATA_OUT],
                 that_system: System[DATA_OUT, DATA_IN],
                 history_length: int = 1) -> Iterator[CONCURRENT_EXAMPLES[INTERACTION_HISTORY, DATA_IN]]:
    history = []
    motor = None

    while True:
        sensor = that_system.react_to(motor)
        motor = this_system.react_to(sensor)

        current_len = len(history)
        if len(history) == history_length:
            yield tuple(history), sensor
        condition = sensor, motor
        history.append(condition)

        for _ in range(current_len - history_length):
            history.pop(0)


if __name__ == "__main__":
    generators_even = tuple((_x for _x in range(0, 100, 2)) for _ in range(10))
    generators_odd = tuple((_x for _x in range(1, 100, 2)) for _ in range(10))

    s = (generators_even[0], generators_odd[0]), (generators_even[1], generators_odd[1])

    g = from_sequences(s)
    for _ in range(55):
        print(next(g))
        pass

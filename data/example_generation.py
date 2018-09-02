#!/usr/bin/env python3
# coding=utf-8

from typing import Iterable, Generator, Tuple, TypeVar, Hashable, List, Any

VALUE_TYPE = TypeVar("VALUE_TYPE", Hashable, Tuple[float, ...])
INPUT = Tuple[VALUE_TYPE, ...]
OUTPUT = Tuple[VALUE_TYPE, ...]
EXAMPLE = Tuple[INPUT[VALUE_TYPE], OUTPUT[VALUE_TYPE]]
EXAMPLE_SEQUENCE = Iterable[EXAMPLE]
JOINT_SEQUENCES = Iterable[Tuple[EXAMPLE, ...]]


def join_sequences(individual_sequences: Iterable[EXAMPLE_SEQUENCE]) -> Generator[Tuple[EXAMPLE, ...], None, None]:
    yield from zip(*individual_sequences)


def example_sequence(source: Iterable[VALUE_TYPE], history_length: int) -> Generator[EXAMPLE, None, None]:
    history: List[VALUE_TYPE] = []
    for each_value in source:
        if len(history) == history_length:
            input_value: INPUT = tuple(history)
            target_value: OUTPUT = (each_value, )
            example: EXAMPLE = (input_value, target_value)
            yield example

        history.append(each_value)
        while history_length < len(history):
            history.pop(0)


def _join_sequences(individual_sequences: Tuple[Iterable[Any], ...]) -> Generator[Tuple[Any, ...], None, None]:
    for current_examples in zip(*individual_sequences):
        yield current_examples


def _join_sequences_2(individual_sequences: Tuple[Iterable[Any], ...]) -> Generator[Tuple[Any, ...], None, None]:
    yield from zip(*individual_sequences)


if __name__ == "__main__":
    s = tuple(list(range(5 * _x, 5 * _x + 3)) for _x in range(1, 4))
    print(s)

    for _x in _join_sequences(s):
        print(_x)

    for _x in _join_sequences_2(s):
        print(_x)

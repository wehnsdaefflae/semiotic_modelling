# coding=utf-8
from typing import Tuple, Iterator, TypeVar


INPUT = TypeVar("INPUT")
TARGET = TypeVar("TARGET")

INPUT_SEQUENCE = Iterator[INPUT]
TARGET_SEQUENCE = Iterator[TARGET]
EXAMPLE_SEQUENCE = Tuple[INPUT_SEQUENCE, TARGET_SEQUENCE]

EXAMPLE = Tuple[INPUT, TARGET]
CONCURRENT_EXAMPLES = Tuple[EXAMPLE, ...]


def from_sequences(sequences: Tuple[EXAMPLE_SEQUENCE, ...]) -> Iterator[CONCURRENT_EXAMPLES]:
    all_sequence_ids = tuple(id(_seq) for _ex in sequences for _seq in _ex)
    if not len(all_sequence_ids) == len(set(all_sequence_ids)):
        raise ValueError("All sequences must be individual iterator instances.")

    no_examples = len(sequences)

    while True:
        examples = tuple((next(_in), next(_tar)) for _in, _tar in sequences)
        if len(examples) != no_examples:
            raise StopIteration()

        yield examples


if __name__ == "__main__":
    generators_even = tuple((_x for _x in range(0, 100, 2)) for _ in range(10))
    generators_odd = tuple((_x for _x in range(1, 100, 2)) for _ in range(10))

    s = (generators_even[0], generators_odd[0]), (generators_even[1], generators_odd[1])

    g = from_sequences(s)
    for _ in range(55):
        print(next(g))
        pass

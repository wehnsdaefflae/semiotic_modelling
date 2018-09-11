#!/usr/bin/env python3
# coding=utf-8
from typing import TypeVar, Tuple, Iterable, Generator

from tools.iterator_split import split_iterator

INPUT_TYPE = TypeVar("INPUT_TYPE")
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")

# base types
EXAMPLE = Tuple[INPUT_TYPE, OUTPUT_TYPE]
CONCURRENT_INPUTS = Tuple[INPUT_TYPE, ...]
CONCURRENT_OUTPUTS = Tuple[OUTPUT_TYPE, ...]

# sequential data_generation formats
PARALLEL_SEQUENCES = Tuple[Iterable[EXAMPLE], ...]
CONCURRENT_EXAMPLES = Iterable[Tuple[EXAMPLE, ...]]
CONCURRENT_IOS = Iterable[Tuple[CONCURRENT_INPUTS, CONCURRENT_OUTPUTS]]


def from_parallel_sequences_to_concurrent_examples(sequences: PARALLEL_SEQUENCES) -> CONCURRENT_EXAMPLES:
    yield from zip(*sequences)


def from_parallel_sequences_to_concurrent_ios(sequences: PARALLEL_SEQUENCES) -> CONCURRENT_IOS:
    yield from from_concurrent_examples_to_concurrent_ios(from_parallel_sequences_to_concurrent_examples(sequences))


def from_concurrent_examples_to_concurrent_ios(examples: CONCURRENT_EXAMPLES) -> CONCURRENT_IOS:
    for concurrent_examples in examples:
        yield tuple(zip(*concurrent_examples))


def from_concurrent_examples_to_parallel_sequences(examples: CONCURRENT_EXAMPLES, no_sequences: int) -> PARALLEL_SEQUENCES:
    def _it() -> Generator[Tuple[EXAMPLE, ...], None, None]:
        for concurrent_examples in examples:
            yield concurrent_examples
    yield from split_iterator(_it(), no_sequences)


def from_concurrent_ios_to_concurrent_examples(ios: CONCURRENT_IOS) -> CONCURRENT_EXAMPLES:
    for concurrent_inputs, concurrent_outputs in ios:
        yield tuple(zip(concurrent_inputs, concurrent_outputs))


def from_concurrent_ios_to_parallel_sequences(ios: CONCURRENT_IOS, no_sequences: int) -> PARALLEL_SEQUENCES:
    yield from from_concurrent_examples_to_parallel_sequences(from_concurrent_ios_to_concurrent_examples(ios), no_sequences)


def main():
    sequences: PARALLEL_SEQUENCES[str, str] = \
        (
            [  # sequence 1
                ("seq1ex1input", "seq1ex1output"), ("seq1ex2input", "seq1ex2output"), ("seq1ex3input", "seq1ex3output")],
            [  # sequence 2
                ("seq2ex1input", "seq2ex1output"), ("seq2ex2input", "seq2ex2output"), ("seq2ex3input", "seq2ex3output")],
            [  # sequence 3
                ("seq3ex1input", "seq3ex1output"), ("seq3ex2input", "seq3ex2output"), ("seq3ex3input", "seq3ex3output")]
        )

    examples: CONCURRENT_EXAMPLES[str, str] = \
        [
            (  # time 1
                ("seq1ex1input", "seq1ex1output"), ("seq2ex1input", "seq2ex1output"), ("seq3ex1input", "seq3ex1output")),
            (  # time 2
                ("seq1ex2input", "seq1ex2output"), ("seq2ex2input", "seq2ex2output"), ("seq3ex2input", "seq3ex2output")),
            (  # time 3
                ("seq1ex3input", "seq1ex3output"), ("seq2ex3input", "seq2ex3output"), ("seq3ex3input", "seq3ex3output"))
        ]

    io_sequence: CONCURRENT_IOS[str, str] = \
        [
            (   # time 1
                (  # inputs 1
                    "seq1ex1input", "seq2ex1input", "seq3ex1input"),
                (  # outputs 1
                    "seq1ex1output", "seq2ex1output", "seq3ex1output")),
            (   # time 2
                (  # inputs 2
                    "seq1ex2input", "seq2ex2input", "seq3ex2input"),
                (  # outputs 2
                    "seq1ex2output", "seq2ex2output", "seq3ex2output")),
            (   # time 3
                (  # inputs 3
                    "seq1ex3input", "seq2ex3input", "seq3ex3input"),
                (  # outputs 3
                    "seq1ex3output", "seq2ex3output", "seq3ex3output"))
        ]

    assert (sequences == tuple(list(_x) for _x in zip(*examples)))
    assert (examples == list(zip(*sequences)))
    assert (io_sequence == [tuple(zip(*each_example)) for each_example in examples])

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


if __name__ == "__main__":
    main()

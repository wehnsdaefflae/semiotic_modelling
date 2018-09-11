# coding=utf-8
from typing import TypeVar, Iterator, Tuple, Collection, Iterable

from data_generation.systems.abstract_classes import System

INPUT = TypeVar("INPUT")
TARGET = TypeVar("TARGET")

SEQUENCE_OF_CONCURRENT_EXAMPLES = Iterator[Tuple[Tuple[INPUT, ...], Tuple[TARGET, ...]]]


ELEMENT = TypeVar("ELEMENT")


def from_sequences(sequences: Collection[Iterable[ELEMENT]]) -> SEQUENCE_OF_CONCURRENT_EXAMPLES[ELEMENT, ELEMENT]:
    ...


DATA_X = TypeVar("DATA_X")
DATA_Y = TypeVar("DATA_Y")


def from_interaction(system_this: System[DATA_X, DATA_Y], system_that: System[DATA_Y, DATA_X]) -> SEQUENCE_OF_CONCURRENT_EXAMPLES[DATA_Y, DATA_X]:
    ...

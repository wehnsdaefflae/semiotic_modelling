# coding=utf-8
from typing import Generic, TypeVar, Tuple, Hashable, Sequence

INPUT_TYPE = TypeVar("INPUT_TYPE")
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")


class Predictor(Generic[INPUT_TYPE, OUTPUT_TYPE]):
    def __init__(self, no_examples: int):
        self.no_examples: int = no_examples
        self.histories = tuple([] for _ in range(no_examples))

    def fit(self, examples: Sequence[Tuple[INPUT_TYPE, OUTPUT_TYPE]]):
        raise NotImplementedError

    def save(self, file_path):
        raise NotImplementedError

    def predict(self, input_values: Sequence[INPUT_TYPE]) -> Tuple[OUTPUT_TYPE, ...]:
        raise NotImplementedError

    def get_structure(self) -> Tuple[int, ...]:
        raise NotImplementedError

    def get_state(self) -> Hashable:
        raise NotImplementedError


class HistoricPredictor(Predictor[Sequence[INPUT_TYPE], OUTPUT_TYPE]):
    def __init__(self, no_examples: int, history_length: int = 1):
        super().__init__(no_examples)

    def fit(self, examples: Sequence[Tuple[INPUT_TYPE, OUTPUT_TYPE]]):
        raise NotImplemented

    def save(self, file_path):
        raise NotImplemented

    def predict(self, input_values: Sequence[INPUT_TYPE]) -> Tuple[OUTPUT_TYPE, ...]:
        raise NotImplemented

    def get_structure(self) -> Tuple[int, ...]:
        raise NotImplemented

    def get_state(self) -> Hashable:
        raise NotImplemented

# coding=utf-8
from typing import Generic, TypeVar, Tuple, Hashable, Sequence

INPUT_TYPE = TypeVar("INPUT_TYPE")
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")

NOMINAL_INPUT = Hashable
NOMINAL_OUTPUT = Hashable
RATIONAL_VECTOR = Tuple[float, ...]


class Predictor(Generic[INPUT_TYPE, OUTPUT_TYPE]):
    def __init__(self, no_examples: int):
        self.no_examples: int = no_examples

    def _fit(self, input_values: Tuple[INPUT_TYPE, ...], target_values: Tuple[OUTPUT_TYPE, ...]):
        raise NotImplementedError

    def fit(self, examples: Sequence[Tuple[INPUT_TYPE, OUTPUT_TYPE]]):
        self.separated_fit(*zip(*examples))     # TODO: if this doesn't work make tuple or list instead

    def separated_fit(self, input_values: Tuple[INPUT_TYPE, ...], target_values: Tuple[OUTPUT_TYPE, ...]):
        assert len(input_values) == len(target_values) == self.no_examples
        self._fit(input_values, target_values)

    def save(self, file_path):
        raise NotImplementedError

    def _predict(self, input_values: Tuple[INPUT_TYPE, ...]) -> Tuple[OUTPUT_TYPE, ...]:
        raise NotImplementedError

    def predict(self, input_values: Tuple[INPUT_TYPE, ...]) -> Tuple[OUTPUT_TYPE, ...]:
        assert len(input_values) == self.no_examples
        output_values = self._predict(input_values)
        assert self.no_examples == len(output_values)
        return output_values

    def get_structure(self) -> Tuple[int, ...]:
        raise NotImplementedError

    def get_state(self) -> Hashable:
        raise NotImplementedError

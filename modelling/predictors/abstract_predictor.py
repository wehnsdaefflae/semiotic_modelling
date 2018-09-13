# coding=utf-8
from typing import Generic, TypeVar, Tuple, Hashable, Sequence

INPUT_TYPE = TypeVar("INPUT_TYPE")
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")

NOMINAL_INPUT = Hashable
NOMINAL_OUTPUT = Hashable
RATIONAL_VECTOR = Tuple[float, ...]


class Predictor(Generic[INPUT_TYPE, OUTPUT_TYPE]):
    def __init__(self, no_examples: int, history_length: int):
        self.no_examples: int = no_examples
        self.histories = tuple([None for _ in range(history_length)] for _ in range(no_examples))
        self.history_length = history_length

    def _fit(self, input_values: Tuple[Tuple[INPUT_TYPE, ...], ...], target_values: Tuple[OUTPUT_TYPE, ...]):
        raise NotImplementedError

    def fit(self, examples: Sequence[Tuple[INPUT_TYPE, OUTPUT_TYPE]]):
        target_values = []
        for _i, (each_input, each_target) in enumerate(examples):
            each_history = self.histories[_i]
            each_history.append(each_input)
            each_history.pop(0)
            target_values.append(each_target)

        historic_condition = tuple(tuple(each_history) for each_history in self.histories)
        self.separated_fit(historic_condition, tuple(target_values))

    def separated_fit(self, input_values: Tuple[Tuple[INPUT_TYPE, ...], ...], target_values: Tuple[OUTPUT_TYPE, ...]):
        assert len(input_values) == len(target_values) == self.no_examples
        self._fit(input_values, target_values)

    def save(self, file_path):
        raise NotImplementedError

    def _predict(self, input_values: Tuple[Tuple[INPUT_TYPE, ...], ...]) -> Tuple[OUTPUT_TYPE, ...]:
        raise NotImplementedError

    def predict(self, input_values: Tuple[INPUT_TYPE, ...]) -> Tuple[OUTPUT_TYPE, ...]:
        assert len(input_values) == self.no_examples
        next_input_values = tuple(tuple(_his[1:]) + _in for _in, _his in zip(input_values, self.histories))

        output_values = self._predict(next_input_values)
        assert self.no_examples == len(output_values)
        return output_values

    def get_structure(self) -> Tuple[int, ...]:
        raise NotImplementedError

    def _get_state(self) -> Hashable:
        raise NotImplementedError

    def get_state(self) -> Tuple[Hashable, Hashable]:
        # TODO: inconsistency in semiotic model: difference trace/state?! integrate again?
        return tuple(tuple(each_history) for each_history in self.histories), self._get_state()

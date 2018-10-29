# coding=utf-8
import numbers
from collections import deque
from typing import TypeVar, Generic, Tuple, Sequence


INPUT_TYPE = TypeVar("INPUT_TYPE")
INPUT_HISTORY = Tuple[INPUT_TYPE, ...]
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")
EXAMPLE_TYPE = Tuple[INPUT_HISTORY, OUTPUT_TYPE]

ANY_TYPE = TypeVar("ANY_TYPE")


class ExampleStream(Generic[INPUT_TYPE, OUTPUT_TYPE]):
    def __init__(self, learn_control: bool, no_examples: int = 1, history_length: int = 1):
        self._learn_control = learn_control
        self._no_examples = no_examples
        self._reward = 0.

        self._history_length = history_length
        self._histories = tuple(deque(maxlen=history_length) for _ in range(no_examples))
        
    def __str__(self):
        raise NotImplementedError()

    def next(self) -> Tuple[EXAMPLE_TYPE, ...]:
        raise NotImplementedError()

    def get_reward(self) -> float:
        return self._reward

    @staticmethod
    def _error(value_output: ANY_TYPE, value_target: ANY_TYPE) -> float:
        if isinstance(value_output, numbers.Rational):
            return abs(value_output - value_target)

        return float(value_output != value_target)

    @staticmethod
    def error(data_output: Sequence[ANY_TYPE], data_target: Sequence[ANY_TYPE]) -> float:
        d = len(data_output)
        assert d == len(data_target)

        error_sum = 0.
        for _o, _t in zip(data_output, data_target):
            if d == 1 and isinstance(_o, str):
                error_sum += ExampleStream._error(_o, _t)

            elif isinstance(_o, Sequence):
                error_sum += ExampleStream.error(_o, _t)

            else:
                error_sum += ExampleStream._error(_o, _t)

        return error_sum / d

# coding=utf-8
from typing import Hashable, Tuple

from _framework.systems_abstract import Predictor

NOMINAL_INPUT = Hashable
NOMINAL_OUTPUT = Hashable

RATIONAL_INPUT = Tuple[float, ...]
RATIONAL_OUTPUT = Tuple[float, ...]


class NominalConstantPredictor(Predictor[NOMINAL_INPUT, NOMINAL_OUTPUT]):
    def __init__(self, no_states: int):
        super().__init__(no_states)
        self._last_output = tuple("#" for _ in range(no_states))
        self._changed = False

    def _react(self, data_in: Tuple[NOMINAL_INPUT, ...]) -> Tuple[NOMINAL_OUTPUT, ...]:
        return self._last_output

    def fit(self, data_in: Tuple[NOMINAL_INPUT, ...], data_out: Tuple[NOMINAL_OUTPUT, ...]):
        if not self._changed:
            self._last_output = data_out
            self._changed = True

    def get_state(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple()


class NominalLastPredictor(Predictor[NOMINAL_INPUT, NOMINAL_OUTPUT]):
    def __init__(self, no_states: int):
        super().__init__(no_states)
        self._last_output = tuple("#" for _ in range(no_states))

    def _react(self, data_in: Tuple[NOMINAL_INPUT, ...]) -> Tuple[NOMINAL_OUTPUT, ...]:
        return self._last_output

    def fit(self, data_in: Tuple[NOMINAL_INPUT, ...], data_out: Tuple[NOMINAL_OUTPUT, ...]):
        self._last_output = data_out

    def get_state(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple()

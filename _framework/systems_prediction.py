# coding=utf-8
from typing import Sequence, Hashable, Tuple

from _framework.systems_abstract import Predictor

NOMINAL_INPUT = Hashable
NOMINAL_OUTPUT = Hashable

RATIONAL_INPUT = Tuple[float, ...]
RATIONAL_OUTPUT = Tuple[float, ...]


class NominalConstantPredictor(Predictor[NOMINAL_INPUT, NOMINAL_OUTPUT]):
    def __init__(self):
        self.last_output = []

    def _react(self, data_in: Sequence[NOMINAL_INPUT]) -> Sequence[NOMINAL_OUTPUT]:
        return tuple(self.last_output)

    def fit(self, data_in: Sequence[NOMINAL_INPUT], data_out: Sequence[NOMINAL_OUTPUT]):
        if len(self.last_output) < 1:
            self.last_output.extend(data_out)


class NominalLastPredictor(Predictor[NOMINAL_INPUT, NOMINAL_OUTPUT]):
    def __init__(self):
        self.last_output = []

    def _react(self, data_in: Sequence[NOMINAL_INPUT]) -> Sequence[NOMINAL_OUTPUT]:
        return tuple(self.last_output)

    def fit(self, data_in: Sequence[NOMINAL_INPUT], data_out: Sequence[NOMINAL_OUTPUT]):
        self.last_output.clear()
        self.last_output.extend(data_out)

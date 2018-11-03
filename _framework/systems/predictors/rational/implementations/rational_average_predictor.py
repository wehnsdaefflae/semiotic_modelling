# coding=utf-8
from typing import Tuple

from _framework.data_types import RATIONAL_INPUT, RATIONAL_OUTPUT, PREDICTOR_STATE
from _framework.systems.predictors.rational.abstract import RationalPredictor
from tools.functionality import smear


class RationalAverage(RationalPredictor):
    def __init__(self, no_states: int, input_dimensions: int, output_dimensions: int, drag: int):
        super().__init__(no_states, input_dimensions, output_dimensions, drag)
        self._average = tuple([0. for _ in range(output_dimensions)] for _ in range(no_states))
        self._iteration = 0

    def __predict(self, data_in: Tuple[RATIONAL_INPUT, ...]) -> Tuple[RATIONAL_OUTPUT, ...]:
        return tuple(tuple(_o) for _o in self._average)

    def __fit(self, data_in: Tuple[RATIONAL_INPUT, ...], data_out: Tuple[RATIONAL_OUTPUT, ...]):
        inertia = self._iteration if self._drag == 0 else self._drag

        for each_target, each_average in zip(data_out, self._average):
            for _i, (_t, _a) in enumerate(zip(each_target, each_average)):
                each_average[_i] = smear(_a, _t, inertia)

    def get_state(self) -> PREDICTOR_STATE:
        return tuple()
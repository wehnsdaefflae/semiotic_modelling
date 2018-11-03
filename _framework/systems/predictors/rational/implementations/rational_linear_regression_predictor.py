# coding=utf-8
from typing import Tuple

from _framework.data_types import RATIONAL_INPUT, RATIONAL_OUTPUT, PREDICTOR_STATE
from _framework.systems.predictors.rational.abstract import RationalPredictor
from tools.regression import Regressor


class RationalLinearRegression(RationalPredictor):
    def __init__(self, no_states: int, input_dimensions: int, output_dimensions: int, drag: int):
        super().__init__(no_states, input_dimensions, output_dimensions, drag)
        self._regressions = tuple(tuple(Regressor(input_dimensions, drag) for _ in range(output_dimensions)) for _ in range(no_states))

    def __predict(self, data_in: Tuple[RATIONAL_INPUT, ...]) -> Tuple[RATIONAL_OUTPUT, ...]:
        output_values = tuple(
            tuple(
                single_regression.output(each_input) for single_regression in each_regression
            ) for each_regression, each_input in zip(self._regressions, data_in)
        )
        return output_values

    def __fit(self, data_in: Tuple[RATIONAL_INPUT, ...], data_out: Tuple[RATIONAL_OUTPUT, ...]):
        for example_index in range(self._no_states):
            each_regression = self._regressions[example_index]

            each_input, each_target = data_in[example_index], data_out[example_index]

            for each_single_regression, each_target_value in zip(each_regression, each_target):
                each_single_regression.fit(each_input, each_target_value)

    def get_state(self) -> PREDICTOR_STATE:
        return tuple()
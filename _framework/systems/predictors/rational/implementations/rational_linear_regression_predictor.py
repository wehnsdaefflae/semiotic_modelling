# coding=utf-8
from typing import Tuple

from _framework.data_types import RATIONAL_INPUT, RATIONAL_OUTPUT, PREDICTOR_STATE
from _framework.systems.predictors.rational.abstract import RationalPredictor
from tools.regression import Regressor


class RationalLinearRegression(RationalPredictor):
    def __init__(self, no_states: int, history_length: int, input_dimensions: int, output_dimensions: int, drag: int):
        super().__init__(no_states, input_dimensions, output_dimensions, drag)
        self._history_length = history_length
        self._regressions = tuple(tuple(Regressor(input_dimensions * history_length, drag) for _ in range(output_dimensions)) for _ in range(no_states))

    def _low_predict(self, data_in: Tuple[Tuple[RATIONAL_INPUT, ...], ...]) -> Tuple[RATIONAL_OUTPUT, ...]:
        data_out = []
        for _regressor_array, _input_history in zip(self._regressions, data_in):
            _flat_input = tuple(_value for _vector in _input_history for _value in _vector)
            _each_output = []
            for _each_regressor in _regressor_array:
                _o = _each_regressor.output(_flat_input)
                _each_output.append(_o)
            data_out.append(tuple(_each_output))
        return tuple(data_out)

    def _low_fit(self, data_in: Tuple[Tuple[RATIONAL_INPUT, ...], ...], data_out: Tuple[RATIONAL_OUTPUT, ...]):
        for _regressor_array, _input_history, _each_target in zip(self._regressions, data_in, data_out):
            _flat_input = tuple(_value for _vector in _input_history for _value in _vector)

            for _each_regressor, _target_value in zip(_regressor_array, _each_target):
                _each_regressor.fit(_flat_input, _target_value)

    def get_state(self) -> PREDICTOR_STATE:
        return tuple()

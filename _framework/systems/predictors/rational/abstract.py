# coding=utf-8
from typing import Tuple

from _framework.data_types import RATIONAL_INPUT, RATIONAL_OUTPUT, PREDICTOR_STATE
from _framework.systems.predictors.abstract import Predictor


class RationalPredictor(Predictor[RATIONAL_INPUT, RATIONAL_OUTPUT]):
    def __init__(self, no_states: int, input_dimensions: int, output_dimensions: int, drag: int):
        super().__init__(no_states)
        assert drag >= 0
        self._in_dim = input_dimensions
        self._out_dim = output_dimensions
        self._drag = drag

    def _low_predict(self, data_in: Tuple[RATIONAL_INPUT, ...]) -> Tuple[RATIONAL_OUTPUT, ...]:
        raise NotImplementedError()

    def _low_fit(self, data_in: Tuple[RATIONAL_INPUT, ...], data_out: Tuple[RATIONAL_OUTPUT, ...]):
        raise NotImplementedError()

    def _predict(self, data_in: Tuple[Tuple[RATIONAL_INPUT, ...], ...]) -> Tuple[RATIONAL_OUTPUT, ...]:
        input_flat = tuple(tuple(_i for _s in _t for _i in _s) for _t in data_in)
        assert all(len(_i) == self._in_dim for _i in input_flat)
        data_out = self._low_predict(input_flat)
        assert all(len(_o) == self._out_dim for _o in data_out)
        return data_out

    def _fit(self, data_in: Tuple[Tuple[RATIONAL_INPUT, ...], ...], data_out: Tuple[RATIONAL_OUTPUT, ...]):
        input_flat = tuple(tuple(_i for _s in _t for _i in _s) for _t in data_in)
        assert all(len(_i) == self._in_dim for _i in input_flat)
        assert all(len(_o) == self._out_dim for _o in data_out)
        self._low_fit(input_flat, data_out)

    def get_state(self) -> PREDICTOR_STATE:
        raise NotImplementedError()

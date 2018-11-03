# coding=utf-8
from typing import Tuple

from _framework.data_types import NOMINAL_INPUT, NOMINAL_OUTPUT, PREDICTOR_STATE
from _framework.systems.predictors.nominal.abstract import NominalPredictor


class NominalLastPredictor(NominalPredictor):
    def __init__(self, no_states: int):
        super().__init__(no_states)
        self._last_output = self._dummy

    def _predict(self, data_in: Tuple[NOMINAL_INPUT, ...]) -> Tuple[NOMINAL_OUTPUT, ...]:
        return self._last_output

    def _fit(self, data_in: Tuple[NOMINAL_INPUT, ...], data_out: Tuple[NOMINAL_OUTPUT, ...]):
        self._last_output = data_out

    def get_state(self) -> PREDICTOR_STATE:
        return tuple()
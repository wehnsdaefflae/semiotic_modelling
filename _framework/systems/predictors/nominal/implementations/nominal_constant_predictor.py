# coding=utf-8
from typing import Tuple

from _framework.data_types import NOMINAL_INPUT, NOMINAL_OUTPUT, PREDICTOR_STATE
from _framework.systems.predictors.nominal.abstract import NominalPredictor


class NominalConstantPredictor(NominalPredictor):
    def __init__(self, no_states: int):
        super().__init__(no_states)
        self._last_output = self._dummy
        self._changed = False

    def _predict(self, data_in: Tuple[Tuple[NOMINAL_INPUT, ...], ...]) -> Tuple[NOMINAL_OUTPUT, ...]:
        return self._last_output

    def _fit(self, data_in: Tuple[Tuple[NOMINAL_INPUT, ...], ...], data_out: Tuple[NOMINAL_OUTPUT, ...]):
        if not self._changed:
            self._last_output = data_out
            self._changed = True

    def get_state(self) -> PREDICTOR_STATE:
        return tuple()
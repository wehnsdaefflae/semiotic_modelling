# coding=utf-8
from typing import Tuple

from _framework.data_types import NOMINAL_INPUT, NOMINAL_OUTPUT, PREDICTOR_STATE
from _framework.systems.predictors.abstract import Predictor


class NominalPredictor(Predictor[NOMINAL_INPUT, NOMINAL_OUTPUT]):
    def __init__(self, no_states: int, *args, **kwargs):
        super().__init__(no_states, *args, **kwargs)
        self._dummy = tuple("#" for _ in range(no_states))

    def _predict(self, data_in: Tuple[Tuple[NOMINAL_INPUT, ...], ...]) -> Tuple[NOMINAL_OUTPUT, ...]:
        raise NotImplementedError()

    def _fit(self, data_in: Tuple[Tuple[NOMINAL_INPUT, ...], ...], data_out: Tuple[NOMINAL_OUTPUT, ...]):
        raise NotImplementedError()

    def get_state(self) -> PREDICTOR_STATE:
        raise NotImplementedError()

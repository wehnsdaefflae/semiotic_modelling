# coding=utf-8
from typing import Tuple, Optional

from _framework.data_types import RATIONAL_INPUT, RATIONAL_OUTPUT, PREDICTOR_STATE
from _framework.systems.predictors.abstract import Predictor


class NominalPredictor(Predictor[RATIONAL_INPUT, RATIONAL_OUTPUT]):
    def __init__(self, no_states: int, *args, **kwargs):
        super().__init__(no_states, *args, **kwargs)
        self._dummy = tuple("#" for _ in range(no_states))

    def react(self, data_in: Optional[Tuple[RATIONAL_INPUT, ...]]) -> Tuple[RATIONAL_OUTPUT, ...]:
        raise NotImplementedError()

    def _fit(self, data_in: Optional[Tuple[RATIONAL_INPUT, ...]], data_out: Tuple[RATIONAL_OUTPUT, ...]):
        raise NotImplementedError()

    def get_state(self) -> PREDICTOR_STATE:
        raise NotImplementedError()

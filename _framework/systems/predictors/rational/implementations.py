# coding=utf-8
from typing import Optional, Tuple

from _framework.data_types import PREDICTOR_STATE, RATIONAL_INPUT, RATIONAL_OUTPUT
from _framework.systems.predictors.rational.abstract import RationalPredictor


class RationalAverage(RationalPredictor):
    def __init__(self, no_states: int, *args, **kwargs):
        super().__init__(no_states, *args, **kwargs)

    def react(self, data_in: Optional[Tuple[RATIONAL_INPUT, ...]]) -> Tuple[RATIONAL_OUTPUT, ...]:
        pass

    def _fit(self, data_in: Optional[Tuple[RATIONAL_INPUT, ...]], data_out: Tuple[RATIONAL_OUTPUT, ...]):
        pass

    def get_state(self) -> PREDICTOR_STATE:
        pass


class RationalLinearRegression(RationalPredictor):
    def __init__(self, no_states: int, *args, **kwargs):
        super().__init__(no_states, *args, **kwargs)

    def react(self, data_in: Optional[Tuple[RATIONAL_INPUT, ...]]) -> Tuple[RATIONAL_OUTPUT, ...]:
        pass

    def _fit(self, data_in: Optional[Tuple[RATIONAL_INPUT, ...]], data_out: Tuple[RATIONAL_OUTPUT, ...]):
        pass

    def get_state(self) -> PREDICTOR_STATE:
        pass


class RationalSemiotic(RationalPredictor):
    def __init__(self, no_states: int, *args, **kwargs):
        super().__init__(no_states, *args, **kwargs)

    def react(self, data_in: Optional[Tuple[RATIONAL_INPUT, ...]]) -> Tuple[RATIONAL_OUTPUT, ...]:
        pass

    def _fit(self, data_in: Optional[Tuple[RATIONAL_INPUT, ...]], data_out: Tuple[RATIONAL_OUTPUT, ...]):
        pass

    def get_state(self) -> PREDICTOR_STATE:
        pass
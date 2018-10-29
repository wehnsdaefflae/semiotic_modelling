# coding=utf-8
from typing import Tuple, TypeVar, Generic, Optional

from _framework.systems.abstract import System
from _framework.data_types import PREDICTOR_STATE

INPUT_TYPE = TypeVar("INPUT_TYPE")
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")


class Predictor(System[Tuple[INPUT_TYPE, ...], Tuple[OUTPUT_TYPE, ...]], Generic[INPUT_TYPE, OUTPUT_TYPE]):
    def __init__(self, no_states: int, *args, **kwargs):
        self.no_states = no_states

    def react(self, data_in: Optional[Tuple[INPUT_TYPE, ...]]) -> Tuple[OUTPUT_TYPE, ...]:
        raise NotImplementedError()

    def _fit(self, data_in: Optional[Tuple[INPUT_TYPE, ...]], data_out: Tuple[OUTPUT_TYPE, ...]):
        raise NotImplementedError()

    def fit(self, data_in: Optional[Tuple[INPUT_TYPE, ...]], data_out: Tuple[OUTPUT_TYPE, ...]):
        assert len(data_out) == self.no_states
        self._fit(data_in, data_out)

    def predict(self, data_in: Optional[Tuple[INPUT_TYPE, ...]]) -> Tuple[OUTPUT_TYPE, ...]:
        return self.react(data_in)

    def get_state(self) -> PREDICTOR_STATE:
        raise NotImplementedError()

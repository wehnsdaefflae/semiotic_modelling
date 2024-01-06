# coding=utf-8
from typing import Tuple, TypeVar, Generic, Optional

from _framework.systems.abstract import System
from _framework.data_types import PREDICTOR_STATE

INPUT_TYPE = TypeVar("INPUT_TYPE")
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")


class Predictor(System[Tuple[Tuple[INPUT_TYPE, ...], ...], Tuple[OUTPUT_TYPE, ...]], Generic[INPUT_TYPE, OUTPUT_TYPE]):
    def __init__(self, no_states: int, *args, **kwargs):
        self._no_states = no_states

    def react(self, data_in: Optional[Tuple[Tuple[INPUT_TYPE, ...], ...]]) -> Tuple[OUTPUT_TYPE, ...]:
        return self.predict(data_in)

    def _fit(self, data_in: Tuple[Tuple[INPUT_TYPE, ...], ...], data_out: Tuple[OUTPUT_TYPE, ...]):
        raise NotImplementedError()

    def fit(self, data_in: Optional[Tuple[Tuple[INPUT_TYPE, ...], ...]], data_out: Tuple[OUTPUT_TYPE, ...]):
        if data_in is None:
            return

        assert len(data_in) == self._no_states
        assert len(data_out) == self._no_states
        self._fit(data_in, data_out)

    def _predict(self, data_in: Tuple[Tuple[INPUT_TYPE, ...], ...]) -> Tuple[OUTPUT_TYPE, ...]:
        raise NotImplementedError()

    def predict(self, data_in: Optional[Tuple[Tuple[INPUT_TYPE, ...], ...]]) -> Tuple[OUTPUT_TYPE, ...]:
        if data_in is None:
            return tuple("#" for _ in range(self._no_states))

        assert len(data_in) == self._no_states
        data_out = self._predict(data_in)
        assert len(data_out) == self._no_states
        return data_out

    def get_state(self) -> PREDICTOR_STATE:
        raise NotImplementedError()

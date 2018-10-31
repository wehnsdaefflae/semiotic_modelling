# coding=utf-8
from typing import Tuple, Optional

from _framework.data_types import NOMINAL_INPUT, NOMINAL_OUTPUT, PREDICTOR_STATE
from _framework.systems.predictors.nominal.abstract import NominalPredictor


class NominalConstantPredictor(NominalPredictor):
    def __init__(self, no_states: int):
        super().__init__(no_states)
        self._last_output = self._dummy
        self._changed = False

    def react(self, data_in: Optional[Tuple[NOMINAL_INPUT, ...]]) -> Tuple[NOMINAL_OUTPUT, ...]:
        return self._last_output

    def _fit(self, data_in: Optional[Tuple[NOMINAL_INPUT, ...]], data_out: Tuple[NOMINAL_OUTPUT, ...]):
        if not self._changed:
            self._last_output = data_out
            self._changed = True

    def get_state(self) -> PREDICTOR_STATE:
        return tuple()


class NominalLastPredictor(NominalPredictor):
    def __init__(self, no_states: int):
        super().__init__(no_states)
        self._last_output = self._dummy

    def react(self, data_in: Optional[Tuple[NOMINAL_INPUT, ...]]) -> Tuple[NOMINAL_OUTPUT, ...]:
        return self._last_output

    def _fit(self, data_in: Optional[Tuple[NOMINAL_INPUT, ...]], data_out: Tuple[NOMINAL_OUTPUT, ...]):
        self._last_output = data_out

    def get_state(self) -> PREDICTOR_STATE:
        return tuple()


class NominalMarkov(NominalPredictor):
    def __init__(self, no_states: int):
        super().__init__(no_states)
        self._table = dict()

    def react(self, data_in: Optional[Tuple[NOMINAL_INPUT, ...]]) -> Tuple[NOMINAL_OUTPUT, ...]:
        sub_dict = self._table.get(data_in)
        if sub_dict is None:
            return self._dummy
        prediction, _ = max(sub_dict.items(), key=lambda x: x[1])
        return prediction

    def _fit(self, data_in: Optional[Tuple[NOMINAL_INPUT, ...]], data_out: Tuple[NOMINAL_OUTPUT, ...]):
        sub_dict = self._table.get(data_in)
        if sub_dict is None:
            self._table[data_in] = {data_out: 1}
        else:
            sub_dict[data_out] = sub_dict.get(data_out, 0) + 1

    def get_state(self) -> PREDICTOR_STATE:
        return tuple()


class NominalSemiotic(NominalPredictor):
    def __init__(self, no_states: int, *args, **kwargs):
        super().__init__(no_states, *args, **kwargs)

    def react(self, data_in: Optional[Tuple[NOMINAL_INPUT, ...]]) -> Tuple[NOMINAL_OUTPUT, ...]:
        pass

    def _fit(self, data_in: Optional[Tuple[NOMINAL_INPUT, ...]], data_out: Tuple[NOMINAL_OUTPUT, ...]):
        pass

    def get_state(self) -> PREDICTOR_STATE:
        pass
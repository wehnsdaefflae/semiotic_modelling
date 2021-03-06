# coding=utf-8
from typing import Callable, Tuple

from _framework.data_types import RATIONAL_INPUT, PREDICTOR_STATE
from _framework.systems.predictors.nominal.resources.semiotic_model import SemioticModel
from _framework.systems.predictors.rational.abstract import RationalPredictor


class RationalSemiotic(RationalPredictor):
    def __init__(self,
                 no_states: int, concrete_history_length: int, input_dimensions: int, output_dimensions: int, drag: int,
                 alpha: int, sigma: float, abstract_history_length: int = 1, fix_level_size_at: Callable[[int], int] = lambda _level: -1):
        super().__init__(no_states, input_dimensions, output_dimensions, drag)
        semiotic_keywords = {
            "is_nominal": False,
            "trace_length": abstract_history_length,
            "fix_level_size_at": fix_level_size_at,
            "input_dimensions": input_dimensions * concrete_history_length,
            "output_dimensions": output_dimensions,
            "drag": 1}
        self._predictor = SemioticModel[Tuple[RATIONAL_INPUT, ...], RATIONAL_INPUT](no_states, alpha, sigma, **semiotic_keywords)

    def _low_predict(self, data_in: Tuple[Tuple[RATIONAL_INPUT, ...], ...]) -> Tuple[RATIONAL_INPUT, ...]:
        flat_input = tuple(tuple(_value for _vector in _input_history for _value in _vector) for _input_history in data_in)
        return self._predictor.predict(flat_input)

    def _low_fit(self, data_in: Tuple[Tuple[RATIONAL_INPUT, ...], ...], data_out: Tuple[RATIONAL_INPUT, ...]):
        flat_input = tuple(tuple(_value for _vector in _input_history for _value in _vector) for _input_history in data_in)
        self._predictor.fit(flat_input, data_out)

    def get_state(self) -> PREDICTOR_STATE:
        return self._predictor.get_state()

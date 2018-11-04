# coding=utf-8
from typing import Callable, Tuple

from _framework.data_types import NOMINAL_INPUT, NOMINAL_OUTPUT, PREDICTOR_STATE
from _framework.systems.predictors.nominal.resources.semiotic_model import SemioticModel
from _framework.systems.predictors.nominal.abstract import NominalPredictor


class NominalSemiotic(NominalPredictor):
    def __init__(self, no_states: int, alpha: int, sigma: float, drag: int = 100, trace_length: int = 1, fix_level_size_at: Callable[[int], int] = lambda _level: -1):
        super().__init__(no_states)
        semiotic_keywords = {
            "is_nominal": True,
            "trace_length": trace_length,
            "fix_level_size_at": fix_level_size_at,
            "input_dimensions": 1,
            "output_dimensions": 1,
            "drag": drag}
        self._predictor = SemioticModel[NOMINAL_INPUT, NOMINAL_OUTPUT](no_states, alpha, sigma, **semiotic_keywords)

    def _predict(self, data_in: Tuple[Tuple[NOMINAL_INPUT, ...], ...]) -> Tuple[NOMINAL_OUTPUT, ...]:
        return self._predictor.predict(data_in)

    def _fit(self, data_in: Tuple[Tuple[NOMINAL_INPUT, ...], ...], data_out: Tuple[NOMINAL_OUTPUT, ...]):
        self._predictor.fit(data_in, data_out)

    def get_state(self) -> PREDICTOR_STATE:
        return self._predictor.get_state()

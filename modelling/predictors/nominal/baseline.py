# coding=utf-8
from typing import Tuple, Hashable, Sequence

from modelling.content import NominalContent
from modelling.predictors.abstract_predictor import Predictor

NOMINAL_INPUT = Hashable
NOMINAL_OUTPUT = Hashable


class NominalMarkovModel(Predictor[NOMINAL_INPUT, NOMINAL_OUTPUT]):
    def __init__(self, no_examples: int):
        super().__init__(no_examples)
        self.models = tuple(NominalContent(0, 0) for _ in range(no_examples))

    def fit(self, examples: Sequence[Tuple[NOMINAL_INPUT, NOMINAL_OUTPUT]]):
        for _i, (each_input, each_target) in enumerate(examples):
            each_model = self.models[_i]
            each_model.adapt(each_input, each_target)

    def save(self, file_path):
        pass

    def predict(self, input_values: Sequence[NOMINAL_INPUT]) -> Tuple[NOMINAL_OUTPUT, ...]:
        return tuple(each_model.predict(input_values[_i]) for _i, each_model in enumerate(self.models))

    def get_structure(self) -> Tuple[int, ...]:
        return 0,

    def get_state(self) -> int:
        return 0

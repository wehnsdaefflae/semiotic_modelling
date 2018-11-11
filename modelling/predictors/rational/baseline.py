# coding=utf-8
from typing import Tuple, Sequence

from modelling.predictors.abstract_predictor import Predictor
from tools.regression_experiments import LinearRegressor


RATIONAL_VECTOR = Tuple[float, ...]


class MovingAverage(Predictor[RATIONAL_VECTOR, RATIONAL_VECTOR]):
    def __init__(self, output_dimension: int, no_examples: int, drag: int):
        super().__init__(no_examples)
        self.output_dimensions = output_dimension
        self.drag = drag
        self.average = tuple([0. for _ in range(output_dimension)] for _ in range(self.no_examples))
        self.initial = True

    def fit(self, examples: Sequence[Tuple[RATIONAL_VECTOR, RATIONAL_VECTOR]]):
        input_values, target_values = zip(*examples)
        if self.initial:
            for each_target, each_average in zip(target_values, self.average):
                for _i, each_target_value in enumerate(each_target):
                    each_average[_i] = each_target_value
            self.initial = False

        else:
            for each_target, each_average in zip(target_values, self.average):
                for _i, each_target_value in enumerate(each_target):
                    each_average[_i] = (each_average[_i] * self.drag + each_target_value) / (self.drag + 1)

    def predict(self, input_values: Sequence[RATIONAL_VECTOR]) -> Tuple[RATIONAL_VECTOR, ...]:
        return tuple(tuple(each_vector) for each_vector in self.average)

    def save(self, file_path):
        raise NotImplementedError

    def get_structure(self) -> Tuple[int, ...]:
        return -1,

    def get_state(self) -> int:
        return 0


class Regression(Predictor[RATIONAL_VECTOR, RATIONAL_VECTOR]):
    def __init__(self, input_dimension: int, output_dimension: int, no_examples: int, drag: int):
        super().__init__(no_examples)
        self.input_dimension = input_dimension
        self.drag = drag
        self.regressions = tuple(tuple(LinearRegressor(input_dimension, self.drag) for _ in range(output_dimension)) for _ in range(no_examples))

    def fit(self, examples: Sequence[Tuple[RATIONAL_VECTOR, RATIONAL_VECTOR]]):
        input_values, target_values = zip(*examples)
        for example_index in range(self.no_examples):
            each_regression = self.regressions[example_index]
            each_input = input_values[example_index]
            each_target = target_values[example_index]
            for each_single_regression, each_target_value in zip(each_regression, each_target):
                each_single_regression.fit(each_input, each_target_value)

    def predict(self, input_values: Sequence[RATIONAL_VECTOR]) -> Tuple[RATIONAL_VECTOR, ...]:
        output_values = tuple(
            tuple(single_regression.output(each_input) for single_regression in each_regression)
            for each_regression, each_input in zip(self.regressions, input_values)
        )
        return output_values

    def save(self, file_path):
        raise NotImplementedError

    def get_structure(self) -> Tuple[int, ...]:
        return -1,

    def get_state(self) -> int:
        return 0

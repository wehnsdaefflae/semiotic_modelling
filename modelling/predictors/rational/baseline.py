# coding=utf-8
from typing import Tuple

from modelling.predictors.abstract_predictor import Predictor, RATIONAL_VECTOR
from tools.regression import Regressor


class MovingAverage(Predictor[RATIONAL_VECTOR, RATIONAL_VECTOR]):
    def __init__(self, output_dimension: int, no_examples: int, drag: int):
        super().__init__(no_examples)
        self.output_dimensions = output_dimension
        self.drag = drag
        self.average = tuple([0. for _ in range(output_dimension)] for _ in range(self.no_examples))
        self.initial = True

    def _fit(self, input_values: Tuple[RATIONAL_VECTOR, ...], target_values: Tuple[RATIONAL_VECTOR, ...]):
        if self.initial:
            for each_target, each_average in zip(target_values, self.average):
                for _i, each_target_value in enumerate(each_target):
                    each_average[_i] = each_target_value
            self.initial = False

        else:
            for each_target, each_average in zip(target_values, self.average):
                for _i, each_target_value in enumerate(each_target):
                    each_average[_i] = (each_average[_i] * self.drag + each_target_value) / (self.drag + 1)

    def _predict(self, input_values: Tuple[RATIONAL_VECTOR, ...]) -> Tuple[RATIONAL_VECTOR, ...]:
        return tuple(tuple(each_average) for each_average in self.average)

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
        self.regressions = tuple(tuple(Regressor(input_dimension, self.drag) for _ in range(output_dimension)) for _ in range(no_examples))

    def _fit(self, input_values: Tuple[RATIONAL_VECTOR, ...], target_values: Tuple[RATIONAL_VECTOR, ...]):
        for example_index in range(self.no_examples):
            each_regression = self.regressions[example_index]
            each_input = input_values[example_index]
            each_target = target_values[example_index]
            for each_single_regression, each_target_value in zip(each_regression, each_target):
                each_single_regression.fit(each_input, each_target_value)

    def _predict(self, input_values: Tuple[RATIONAL_VECTOR, ...]) -> Tuple[RATIONAL_VECTOR, ...]:
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

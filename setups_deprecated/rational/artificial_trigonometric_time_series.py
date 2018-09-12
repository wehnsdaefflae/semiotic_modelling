# coding=utf-8
import os

from matplotlib import pyplot

from data_generation.deprecated_data_types import from_parallel_sequences_to_concurrent_examples
from environments.non_interactive import examples_rational_trigonometric
from setups_deprecated.evaluations import prediction
from modelling.predictors.rational.semiotic import RationalSemioticModel
from modelling.predictors.rational.baseline import Regression


def _artificial_isolated(iterations: int):
    sequences = examples_rational_trigonometric(history_length=1),

    predictor_a = Regression(input_dimension=1, output_dimension=1, no_examples=len(sequences), drag=100)
    predictor_b = RationalSemioticModel(input_dimension=1, output_dimension=1, no_examples=len(sequences), alpha=10, sigma=.5, drag=100,
                                        trace_length=1)
    predictors = predictor_a, predictor_b

    examples = from_parallel_sequences_to_concurrent_examples(sequences)
    prediction(examples, predictors, iterations=iterations, rational=True)


def _artificial_synthesis(iterations: int):
    offset = 2735
    sequence_a = examples_rational_trigonometric()
    sequence_b = examples_rational_trigonometric()
    for _ in range(offset):
        next(sequence_b)
    sequences = sequence_a, sequence_b

    predictor_a = Regression(input_dimension=1, output_dimension=1, no_examples=len(sequences), drag=100)
    predictor_b = RationalSemioticModel(input_dimension=1, output_dimension=1, no_examples=len(sequences), alpha=10, sigma=.5, drag=100,
                                        trace_length=1)
    predictors = predictor_a, predictor_b

    examples = from_parallel_sequences_to_concurrent_examples(sequences)
    prediction(examples, predictors, iterations=iterations, rational=True)


if __name__ == "__main__":
    duration = 50000

    pyplot.suptitle(os.path.basename(__file__))

    _artificial_isolated(duration)
    _artificial_synthesis(duration)

    pyplot.show()



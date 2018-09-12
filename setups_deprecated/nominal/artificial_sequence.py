#!/usr/bin/env python3
# coding=utf-8
import os

from matplotlib import pyplot

from data_generation.deprecated_data_types import from_parallel_sequences_to_concurrent_examples
from data_generation.deprecated_example_generation import example_sequence
from environments.non_interactive import sequence_nominal_alternating
from setups_deprecated.evaluations import prediction
from modelling.predictors.nominal.semiotic import NominalSemioticModel
from modelling.predictors.nominal.baseline import NominalMarkovModel


def _artificial_isolated(iterations: int):
    sequences = example_sequence(sequence_nominal_alternating(), history_length=1),

    predictor_a = NominalMarkovModel(no_examples=len(sequences))
    predictor_b = NominalSemioticModel(no_examples=len(sequences), alpha=0, sigma=1., trace_length=1)
    predictors = predictor_a, predictor_b

    examples = from_parallel_sequences_to_concurrent_examples(sequences)
    prediction(examples, predictors, rational=False, iterations=iterations)


def _artificial_synthesis(iterations: int):
    sequence_a = example_sequence(sequence_nominal_alternating(), history_length=1)
    sequence_b = example_sequence(sequence_nominal_alternating(), history_length=1)
    sequences = sequence_a, sequence_b

    predictor_a = NominalMarkovModel(no_examples=len(sequences))
    predictor_b = NominalSemioticModel(no_examples=len(sequences), alpha=0, sigma=1., trace_length=1)
    predictors = predictor_a, predictor_b

    examples = from_parallel_sequences_to_concurrent_examples(sequences)
    prediction(examples, predictors, iterations=iterations, rational=False)


if __name__ == "__main__":
    duration = 50000

    pyplot.suptitle(os.path.basename(__file__))

    _artificial_isolated(duration)
    _artificial_synthesis(duration)

    pyplot.show()

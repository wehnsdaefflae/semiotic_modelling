#!/usr/bin/env python3
# coding=utf-8
import os

from matplotlib import pyplot

from data_generation.deprecated_data_types import from_parallel_sequences_to_concurrent_examples
from data_generation.deprecated_example_generation import example_sequence
from environments.non_interactive import sequence_nominal_text
from setups_deprecated.evaluations import prediction
from modelling.predictors.nominal.semiotic import NominalSemioticModel
from modelling.predictors.nominal.baseline import NominalMarkovModel
from tools.load_configs import Config


def _natural_isolated(iterations: int):
    c = Config("../../configs/config.json")
    sequences = example_sequence(sequence_nominal_text(c["data_dir"] + "Texts/pride_prejudice.txt"), history_length=1),

    predictor_a = NominalMarkovModel(no_examples=len(sequences))
    predictor_b = NominalSemioticModel(no_examples=len(sequences), alpha=50, sigma=.1, history_length=1)
    predictors = predictor_a, predictor_b

    examples = from_parallel_sequences_to_concurrent_examples(sequences)
    prediction(examples, predictors, iterations=iterations, rational=False)


def _natural_synthesis(iterations: int):
    c = Config("../../configs/config.json")
    sequence_a = example_sequence(sequence_nominal_text(c["data_dir"] + "Texts/pride_prejudice.txt"), history_length=1)
    sequence_b = example_sequence(sequence_nominal_text(c["data_dir"] + "Texts/mansfield_park.txt"), history_length=1)
    sequence = sequence_a, sequence_b

    predictor_a = NominalMarkovModel(no_examples=len(sequence))
    predictor_b = NominalSemioticModel(no_examples=len(sequence), alpha=50, sigma=.1, history_length=1)
    predictors = predictor_a, predictor_b

    examples = from_parallel_sequences_to_concurrent_examples(sequence)
    prediction(examples, predictors, iterations=iterations, rational=False)


if __name__ == "__main__":
    duration = 500000

    pyplot.suptitle(os.path.basename(__file__))

    _natural_isolated(duration)
    _natural_synthesis(duration)

    pyplot.show()

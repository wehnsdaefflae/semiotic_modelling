#!/usr/bin/env python3
# coding=utf-8

from matplotlib import pyplot

from data.example_generation import example_sequence, join_sequences
from environments.non_interactive import sequence_nominal_text, sequence_nominal_alternating
from evaluations.experiments import experiment
from modelling.model_types.nominal.semiotic import NominalSemioticModel
from modelling.model_types.nominal.baseline import NominalMarkovModelIsolated, NominalMarkovModelIntegrated
from tools.load_configs import Config


def _artificial_isolated(iterations: int):
    sequences = example_sequence(sequence_nominal_alternating(), history_length=1),

    predictor_a = NominalMarkovModelIsolated(no_examples=len(sequences))
    predictor_b = NominalSemioticModel(no_examples=len(sequences), alpha=0, sigma=1., trace_length=1)
    predictors = predictor_a, predictor_b

    examples = join_sequences(sequences)
    experiment(examples, predictors, rational=False, iterations=iterations)


def _natural_isolated(iterations: int):
    c = Config("../configs/config.json")
    sequences = example_sequence(sequence_nominal_text(c["data_dir"] + "Texts/pride_prejudice.txt"), history_length=1),

    predictor_a = NominalMarkovModelIsolated(no_examples=len(sequences))
    predictor_b = NominalSemioticModel(no_examples=len(sequences), alpha=50, sigma=.1, trace_length=1)
    predictors = predictor_a, predictor_b

    examples = join_sequences(sequences)
    experiment(examples, predictors, iterations=iterations, rational=False)


def _artificial_synthesis(iterations: int):
    sequence_a = example_sequence(sequence_nominal_alternating(), history_length=1)
    sequence_b = example_sequence(sequence_nominal_alternating(), history_length=1)
    sequences = sequence_a, sequence_b

    predictor_a = NominalMarkovModelIntegrated(no_examples=len(sequences))
    predictor_b = NominalSemioticModel(no_examples=len(sequences), alpha=0, sigma=1., trace_length=1)
    predictors = predictor_a, predictor_b

    examples = join_sequences(sequences)
    experiment(examples, predictors, iterations=iterations, rational=False)


def _natural_synthesis(iterations: int):
    c = Config("../configs/config.json")
    sequence_a = example_sequence(sequence_nominal_text(c["data_dir"] + "Texts/pride_prejudice.txt"), history_length=1)
    sequence_b = example_sequence(sequence_nominal_text(c["data_dir"] + "Texts/mansfield_park.txt"), history_length=1)
    sequence = sequence_a, sequence_b

    predictor_a = NominalMarkovModelIntegrated(no_examples=len(sequence))
    predictor_b = NominalSemioticModel(no_examples=len(sequence), alpha=50, sigma=.1, trace_length=1)
    predictors = predictor_a, predictor_b

    examples = join_sequences(sequence)
    experiment(examples, predictors, iterations=iterations, rational=False)


def artificial_experiment(iterations: int = 50000):
    _artificial_isolated(iterations)
    _artificial_synthesis(iterations)

    pyplot.legend()
    pyplot.show()


def natural_experiment(iterations: int = 500000):
    _natural_isolated(iterations)
    _natural_synthesis(iterations)

    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    # TODO: translate nominal into rational environments (generator!)
    natural_experiment()
    # artificial_experiment()

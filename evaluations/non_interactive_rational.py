# coding=utf-8
from matplotlib import pyplot

from data.example_generation import example_sequence, join_sequences
from environments.non_interactive import examples_rational_trigonometric, sequence_rational_crypto
from evaluations.experiments import experiment, experiment
from modelling.model_types import RegressionIsolated, RationalSemioticModel, RegressionIntegrated
from tools.load_configs import Config


def _artificial_isolated(iterations: int):
    sequences = examples_rational_trigonometric(history_length=1),

    predictor_a = RegressionIsolated(input_dimension=1, output_dimension=1, no_examples=len(sequences), drag=100)
    predictor_b = RationalSemioticModel(input_dimensions=1, output_dimensions=1, no_examples=len(sequences), alpha=10, sigma=.5, drag=100,
                                        trace_length=1)
    predictors = predictor_a, predictor_b

    examples = join_sequences(sequences)
    experiment(examples, predictors, iterations=iterations, rational=True)


def _natural_isolated(iterations: int):
    c = Config("../configs/config.json")
    sequence_a = sequence_rational_crypto(c["data_dir"] + "binance/EOSETH.csv", 60, start_val=1501113780, end_val=1529712000)
    sequences = example_sequence(sequence_a, history_length=1),

    predictor_a = RegressionIsolated(input_dimension=1, output_dimension=1, no_examples=len(sequences), drag=100)
    predictor_b = RationalSemioticModel(input_dimensions=1, output_dimensions=1, no_examples=len(sequences), alpha=10, sigma=.8, drag=100,
                                        trace_length=1)
    predictors = predictor_a, predictor_b

    examples = join_sequences(sequences)
    experiment(examples, predictors, iterations=iterations, rational=True)


def _artificial_synthesis(iterations: int):
    offset = 2735
    sequence_a = examples_rational_trigonometric()
    sequence_b = examples_rational_trigonometric()
    for _ in range(offset):
        next(sequence_b)
    sequences = sequence_a, sequence_b

    predictor_a = RegressionIntegrated(input_dimension=1, output_dimension=1, no_examples=len(sequences), drag=100)
    predictor_b = RationalSemioticModel(input_dimensions=1, output_dimensions=1, no_examples=len(sequences), alpha=10, sigma=.5, drag=100,
                                        trace_length=1)
    predictors = predictor_a, predictor_b

    examples = join_sequences(sequences)
    experiment(examples, predictors, iterations=iterations, rational=True)


def _natural_synthesis(iterations: int):
    c = Config("../configs/config.json")
    sequence_a = sequence_rational_crypto(c["data_dir"] + "binance/EOSETH.csv", 60, start_val=1501113780, end_val=1529712000)
    sequence_b = sequence_rational_crypto(c["data_dir"] + "binance/QTUMETH.csv", 60, start_val=1501113780, end_val=1529712000)
    # SNT
    sequences = example_sequence(sequence_a, history_length=1), example_sequence(sequence_b, history_length=1)

    predictor_a = RegressionIntegrated(input_dimension=1, output_dimension=1, no_examples=len(sequences), drag=100)
    predictor_b = RationalSemioticModel(input_dimensions=1, output_dimensions=1, no_examples=len(sequences), alpha=10, sigma=.8, drag=100,
                                        trace_length=1)
    predictors = predictor_a, predictor_b

    examples = join_sequences(sequences)
    experiment(examples, predictors, iterations=iterations, rational=True)


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
    natural_experiment()
    # artificial_experiment()

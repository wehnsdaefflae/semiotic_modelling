# coding=utf-8
from matplotlib import pyplot

from data.data_types import from_parallel_sequences_to_concurrent_examples
from data.example_generation import example_sequence
from environments.non_interactive import sequence_rational_crypto
from evaluations.experiments import experiment
from modelling.predictors.rational.semiotic import RationalSemioticModel
from modelling.predictors.rational.baseline import RegressionIsolated, RegressionIntegrated
from tools.load_configs import Config


def _natural_isolated(iterations: int):
    c = Config("../../configs/config.json")
    sequence_a = sequence_rational_crypto(c["data_dir"] + "binance/EOSETH.csv", 60, start_val=1501113780, end_val=1529712000)
    sequences = example_sequence(sequence_a, history_length=1),

    predictor_a = RegressionIsolated(input_dimension=1, output_dimension=1, no_examples=len(sequences), drag=100)
    predictor_b = RationalSemioticModel(input_dimensions=1, output_dimensions=1, no_examples=len(sequences), alpha=10, sigma=.8, drag=100,
                                        trace_length=1)
    predictors = predictor_a, predictor_b

    examples = from_parallel_sequences_to_concurrent_examples(sequences)
    experiment(examples, predictors, iterations=iterations, rational=True)


def _natural_synthesis(iterations: int):
    c = Config("../../configs/config.json")
    sequence_a = sequence_rational_crypto(c["data_dir"] + "binance/EOSETH.csv", 60, start_val=1501113780, end_val=1529712000)
    sequence_b = sequence_rational_crypto(c["data_dir"] + "binance/QTUMETH.csv", 60, start_val=1501113780, end_val=1529712000)
    # SNT
    sequences = example_sequence(sequence_a, history_length=1), example_sequence(sequence_b, history_length=1)

    predictor_a = RegressionIntegrated(input_dimension=1, output_dimension=1, no_examples=len(sequences), drag=100)
    predictor_b = RationalSemioticModel(input_dimensions=1, output_dimensions=1, no_examples=len(sequences), alpha=10, sigma=.8, drag=100,
                                        trace_length=1)
    predictors = predictor_a, predictor_b

    examples = from_parallel_sequences_to_concurrent_examples(sequences)
    experiment(examples, predictors, iterations=iterations, rational=True)


if __name__ == "__main__":
    duration = 500000

    _natural_isolated(duration)
    _natural_synthesis(duration)

    pyplot.tight_layout()
    pyplot.show()

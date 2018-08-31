from matplotlib import pyplot

from data.data_processing import examples_from_sequence
from environments.non_interactive import examples_rational_trigonometric, sequence_rational_crypto
from evaluations.experiments import experiment_non_interactive
from modelling.model_types import RegressionIsolated, RationalSemioticModel, RegressionIntegrated
from tools.load_configs import Config


def _artificial_isolated(iterations: int):
    environments = [examples_rational_trigonometric(history_length=1)]

    predictor_a = RegressionIsolated(input_dimension=1, output_dimension=1, no_examples=len(environments), drag=100)
    predictor_b = RationalSemioticModel(input_dimensions=1, output_dimensions=1, no_examples=len(environments), alpha=10, sigma=.5, drag=100,
                                        trace_length=1)
    predictors = [predictor_a, predictor_b]

    experiment_non_interactive(environments, predictors, iterations=iterations, rational=True)


def _natural_isolated(iterations: int):
    c = Config("../configs/config.json")
    env = sequence_rational_crypto(c["data_dir"] + "binance/EOSETH.csv", 60, start_val=1501113780, end_val=1529712000)
    environments = [examples_from_sequence(env, history_length=1)]

    predictor_a = RegressionIsolated(input_dimension=1, output_dimension=1, no_examples=len(environments), drag=100)
    predictor_b = RationalSemioticModel(input_dimensions=1, output_dimensions=1, no_examples=len(environments), alpha=10, sigma=.8, drag=100,
                                        trace_length=1)
    predictors = [predictor_a, predictor_b]

    experiment_non_interactive(environments, predictors, iterations=iterations, rational=True)


def _artificial_transfer(iterations: int):
    offset = 2735
    environment_a = examples_rational_trigonometric()
    environment_b = examples_rational_trigonometric()
    for _ in range(offset):
        next(environment_b)
    environments = [environment_a, environment_b]

    predictor_a = RegressionIntegrated(input_dimension=1, output_dimension=1, no_examples=len(environments), drag=100)
    predictor_b = RationalSemioticModel(input_dimensions=1, output_dimensions=1, no_examples=len(environments), alpha=10, sigma=.5, drag=100,
                                        trace_length=1)
    predictors = [predictor_a, predictor_b]

    experiment_non_interactive(environments, predictors, iterations=iterations, rational=True)


def _natural_transfer(iterations: int):
    c = Config("../configs/config.json")
    environment_a = sequence_rational_crypto(c["data_dir"] + "binance/EOSETH.csv", 60, start_val=1501113780, end_val=1529712000)
    environment_b = sequence_rational_crypto(c["data_dir"] + "binance/QTUMETH.csv", 60, start_val=1501113780, end_val=1529712000)
    # SNT
    environments = [examples_from_sequence(environment_a, history_length=1), examples_from_sequence(environment_b, history_length=1)]

    predictor_a = RegressionIntegrated(input_dimension=1, output_dimension=1, no_examples=len(environments), drag=100)
    predictor_b = RationalSemioticModel(input_dimensions=1, output_dimensions=1, no_examples=len(environments), alpha=10, sigma=.8, drag=100,
                                        trace_length=1)
    predictors = [predictor_a, predictor_b]

    experiment_non_interactive(environments, predictors, iterations=iterations, rational=True)


def artificial_experiment(iterations: int = 50000):
    _artificial_isolated(iterations)
    _artificial_transfer(iterations)

    pyplot.legend()
    pyplot.show()


def natural_experiment(iterations: int = 500000):
    _natural_isolated(iterations)
    _natural_transfer(iterations)

    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    natural_experiment()
    # artificial_experiment()

from matplotlib import pyplot

from data.example_generation import example_sequence
from environments.non_interactive import sequence_nominal_text, sequence_nominal_alternating
from evaluations.experiments import experiment_non_interactive
from modelling.model_types import NominalSemioticModel, NominalMarkovModelIsolated, NominalMarkovModelIntegrated
from tools.load_configs import Config


def _artificial_isolated(iterations: int):
    environments = [example_sequence(sequence_nominal_alternating(), history_length=1)]

    predictor_a = NominalMarkovModelIsolated(no_examples=len(environments))
    predictor_b = NominalSemioticModel(no_examples=len(environments), alpha=0, sigma=1., trace_length=1)
    predictors = [predictor_a, predictor_b]

    experiment_non_interactive(environments, predictors, rational=False, iterations=iterations)


def _natural_isolated(iterations: int):
    c = Config("../configs/config.json")
    environments = [example_sequence(sequence_nominal_text(c["data_dir"] + "Texts/pride_prejudice.txt"), history_length=1)]

    predictor_a = NominalMarkovModelIsolated(no_examples=len(environments))
    predictor_b = NominalSemioticModel(no_examples=len(environments), alpha=50, sigma=.1, trace_length=1)
    predictors = [predictor_a, predictor_b]

    experiment_non_interactive(environments, predictors, iterations=iterations, rational=False)


def _artificial_transfer(iterations: int):
    environment_a = example_sequence(sequence_nominal_alternating(), history_length=1)
    environment_b = example_sequence(sequence_nominal_alternating(), history_length=1)
    environments = [environment_a, environment_b]

    predictor_a = NominalMarkovModelIntegrated(no_examples=len(environments))
    predictor_b = NominalSemioticModel(no_examples=len(environments), alpha=0, sigma=1., trace_length=1)
    predictors = [predictor_a, predictor_b]

    experiment_non_interactive(environments, predictors, iterations=iterations, rational=False)


def _natural_transfer(iterations: int):
    c = Config("../configs/config.json")
    environment_a = example_sequence(sequence_nominal_text(c["data_dir"] + "Texts/pride_prejudice.txt"), history_length=1)
    environment_b = example_sequence(sequence_nominal_text(c["data_dir"] + "Texts/mansfield_park.txt"), history_length=1)
    environments = [environment_a, environment_b]

    predictor_a = NominalMarkovModelIntegrated(no_examples=len(environments))
    predictor_b = NominalSemioticModel(no_examples=len(environments), alpha=50, sigma=.1, trace_length=1)
    predictors = [predictor_a, predictor_b]

    experiment_non_interactive(environments, predictors, iterations=iterations, rational=False)


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
    # TODO: translate nominal into rational environments (generator!)
    natural_experiment()
    # artificial_experiment()

from matplotlib import pyplot

from data.data_processing import examples_from_sequence
from environments.functionality import prediction_functionality
from environments.non_interactive import sequence_nominal_text, sequence_nominal_alternating
from evaluations.experiments import experiment_non_interactive
from modelling.model_types import NominalSemioticModel, NominalMarkovModelIsolated, NominalMarkovModelIntegrated
from tools.load_configs import Config


def _artificial_isolated(iterations: int):
    environments = [examples_from_sequence(sequence_nominal_alternating(), history_length=1)]
    predictor = NominalMarkovModelIsolated(no_examples=len(environments))
    experiment_non_interactive(environments, predictor, rational=False, iterations=iterations)

    environments = [examples_from_sequence(sequence_nominal_alternating(), history_length=1)]
    predictor = NominalSemioticModel(no_examples=len(environments), alpha=0, sigma=1., trace_length=1)
    experiment_non_interactive(environments, predictor, rational=False, iterations=iterations)


def _natural_isolated(iterations: int):
    c = Config("../configs/config.json")

    environments = [examples_from_sequence(sequence_nominal_text(c["data_dir"] + "Texts/pride_prejudice.txt"), history_length=1)]
    predictor = NominalMarkovModelIsolated(no_examples=len(environments))
    experiment_non_interactive(environments, predictor, iterations=iterations, rational=False)

    environments = [examples_from_sequence(sequence_nominal_text(c["data_dir"] + "Texts/pride_prejudice.txt"), history_length=1)]
    predictor = NominalSemioticModel(no_examples=len(environments), alpha=50, sigma=.1, trace_length=1)
    experiment_non_interactive(environments, predictor, iterations=iterations, rational=False)


def _artificial_transfer(iterations: int):
    environment_a = examples_from_sequence(sequence_nominal_alternating(), history_length=1)
    environment_b = examples_from_sequence(sequence_nominal_alternating(), history_length=1)
    environments = [environment_a, environment_b]
    predictor = NominalMarkovModelIntegrated(no_examples=len(environments))
    experiment_non_interactive(environments, predictor, iterations=iterations, rational=False)

    environment_a = examples_from_sequence(sequence_nominal_alternating(), history_length=1)
    environment_b = examples_from_sequence(sequence_nominal_alternating(), history_length=1)
    environments = [environment_a, environment_b]
    predictor = NominalSemioticModel(no_examples=len(environments), alpha=0, sigma=1., trace_length=1)
    experiment_non_interactive(environments, predictor, iterations=iterations, rational=False)


def _natural_transfer(iterations: int):
    c = Config("../configs/config.json")

    environment_a = examples_from_sequence(sequence_nominal_text(c["data_dir"] + "Texts/pride_prejudice.txt"), history_length=1)
    environment_b = examples_from_sequence(sequence_nominal_text(c["data_dir"] + "Texts/mansfield_park.txt"), history_length=1)
    environments = [environment_a, environment_b]
    predictor = NominalMarkovModelIntegrated(no_examples=len(environments))
    experiment_non_interactive(environments, predictor, iterations=iterations, rational=False)

    environment_a = examples_from_sequence(sequence_nominal_text(c["data_dir"] + "Texts/pride_prejudice.txt"), history_length=1)
    environment_b = examples_from_sequence(sequence_nominal_text(c["data_dir"] + "Texts/mansfield_park.txt"), history_length=1)
    environments = [environment_a, environment_b]
    predictor = NominalSemioticModel(no_examples=len(environments), alpha=50, sigma=.1, trace_length=1)
    experiment_non_interactive(environments, predictor, iterations=iterations, rational=False)


def artificial_experiment(iterations: int = 50000):
    f = prediction_functionality(sequence_nominal_alternating(), iterations, rational=False)
    print("example sequence functionality: {:05.3f}".format(f))

    _artificial_isolated(iterations)
    _artificial_transfer(iterations)

    pyplot.legend()
    pyplot.show()


def natural_experiment(iterations: int = 500000):
    c = Config("../configs/config.json")

    f = prediction_functionality(sequence_nominal_text(c["data_dir"] + "Texts/pride_prejudice.txt"), iterations, rational=False)
    print("sequence a functionality: {:05.3f}".format(f))
    f = prediction_functionality(sequence_nominal_text(c["data_dir"] + "Texts/mansfield_park.txt"), iterations, rational=False)
    print("sequence b functionality: {:05.3f}".format(f))

    _natural_isolated(iterations)
    _natural_transfer(iterations)

    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    # natural_experiment()
    artificial_experiment()

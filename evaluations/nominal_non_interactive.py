from matplotlib import pyplot

from environments.functionality import prediction_functionality
from environments.non_interactive import env_text, env_ascending_descending_nominal
from evaluations.experiments_non_interactive import one_step_prediction
from modelling.model_types import NominalSemioticModel, NominalMarkovModel
from tools.load_configs import Config


def _artificial_isolated(iterations: int):
    environments = [env_ascending_descending_nominal()]
    predictor = NominalMarkovModel(no_examples=len(environments))
    one_step_prediction(environments, predictor, iterations=iterations, rational=False)

    environments = [env_ascending_descending_nominal()]
    predictor = NominalSemioticModel(no_examples=len(environments), alpha=0, sigma=1., trace_length=1)
    one_step_prediction(environments, predictor, iterations=iterations, rational=False)


def _natural_isolated(iterations: int):
    c = Config("../configs/config.json")

    environments = [env_text(c["data_dir"] + "Texts/pride_prejudice.txt")]
    predictor = NominalMarkovModel(no_examples=len(environments))
    one_step_prediction(environments, predictor, iterations=iterations, rational=False, history_length=1)

    environments = [env_text(c["data_dir"] + "Texts/pride_prejudice.txt")]
    predictor = NominalSemioticModel(no_examples=len(environments), alpha=50, sigma=.1, trace_length=1)
    one_step_prediction(environments, predictor, iterations=iterations, rational=False, history_length=1)


def _artificial_transfer(iterations: int):
    environment_a = env_ascending_descending_nominal()
    environment_b = env_ascending_descending_nominal()
    environments = [environment_a, environment_b]
    predictor = NominalSemioticModel(no_examples=len(environments), alpha=0, sigma=1., trace_length=1)

    one_step_prediction(environments, predictor, iterations=iterations, rational=False)


def _natural_transfer(iterations: int):
    c = Config("../configs/config.json")

    environment_a = env_text(c["data_dir"] + "Texts/pride_prejudice.txt")
    environment_b = env_text(c["data_dir"] + "Texts/mansfield_park.txt")
    environments = [environment_a, environment_b]
    predictor = NominalSemioticModel(no_examples=len(environments), alpha=50, sigma=.1, trace_length=1)

    one_step_prediction(environments, predictor, iterations=iterations, rational=False)


def artificial_experiment(iterations: int = 50000):
    f = prediction_functionality(env_ascending_descending_nominal(), iterations, rational=False)
    print("example sequence functionality: {:05.3f}".format(f))

    _artificial_isolated(iterations)
    _artificial_transfer(iterations)

    pyplot.legend()
    pyplot.show()


def natural_experiment(iterations: int = 500000):
    c = Config("../configs/config.json")

    f = prediction_functionality(env_text(c["data_dir"] + "Texts/pride_prejudice.txt"), iterations, rational=False)
    print("sequence a functionality: {:05.3f}".format(f))
    f = prediction_functionality(env_text(c["data_dir"] + "Texts/mansfield_park.txt"), iterations, rational=False)
    print("sequence b functionality: {:05.3f}".format(f))

    _natural_isolated(iterations)
    _natural_transfer(iterations)

    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    natural_experiment()
    # artificial_experiment()

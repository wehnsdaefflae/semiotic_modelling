from matplotlib import pyplot

from environments.functionality import generic_functionality, prediction_functionality
from environments.non_interactive import env_trigonometric_rational, env_crypto, env_ascending_descending_nominal
from evaluations.experiments_non_interactive import one_step_prediction, example_prediction
from modelling.model_types import Regression, RationalSemioticModel
from tools.load_configs import Config


def _artificial_isolated(iterations: int):
    environments = [env_trigonometric_rational()]
    predictor = Regression(input_dimension=1, output_dimension=1, no_examples=len(environments), drag=100)
    example_prediction(environments, predictor, iterations=iterations, rational=True, history_length=1)

    environments = [env_trigonometric_rational()]
    predictor = RationalSemioticModel(input_dimensions=1, output_dimensions=1, no_examples=len(environments),
                                      alpha=10, sigma=.8, drag=100, trace_length=1)
    example_prediction(environments, predictor, iterations=iterations, rational=True, history_length=1)


def _natural_isolated(iterations: int):
    c = Config("../configs/config.json")

    environments = [env_crypto(c["data_dir"] + "23Jun2017-23Jun2018-1m/EOSETH.csv", 60, start_val=1501113780, end_val=1529712000)]
    predictor = Regression(input_dimension=1, output_dimension=1, no_examples=len(environments), drag=100)
    one_step_prediction(environments, predictor, iterations=iterations, rational=True, history_length=1)

    environments = [env_crypto(c["data_dir"] + "23Jun2017-23Jun2018-1m/EOSETH.csv", 60, start_val=1501113780, end_val=1529712000)]
    predictor = RationalSemioticModel(input_dimensions=1, output_dimensions=1, no_examples=len(environments),
                                      alpha=10, sigma=.8, drag=100, trace_length=1)
    one_step_prediction(environments, predictor, iterations=iterations, rational=True, history_length=1)


def _artificial_transfer(iterations: int):
    environment_a = env_trigonometric_rational()
    environment_b = env_trigonometric_rational()
    for _ in range(2735):
        next(environment_b)
    environments = [environment_a, environment_b]
    predictor = RationalSemioticModel(input_dimensions=1, output_dimensions=1, no_examples=len(environments),
                                      alpha=10, sigma=.8, drag=100, trace_length=1)

    example_prediction(environments, predictor, iterations=iterations, rational=True, history_length=1)


def _natural_transfer(iterations: int):
    c = Config("../configs/config.json")

    environment_a = env_crypto(c["data_dir"] + "23Jun2017-23Jun2018-1m/EOSETH.csv", 60, start_val=1501113780, end_val=1529712000)
    environment_b = env_crypto(c["data_dir"] + "23Jun2017-23Jun2018-1m/QTUMETH.csv", 60, start_val=1501113780, end_val=1529712000)
    # SNT
    environments = [environment_a, environment_b]
    predictor = RationalSemioticModel(input_dimensions=1, output_dimensions=1, no_examples=len(environments),
                                      alpha=10, sigma=.8, drag=100, trace_length=1)

    one_step_prediction(environments, predictor, iterations=iterations, rational=True, history_length=1)


def artificial_experiment(iterations: int = 50000):
    f = generic_functionality(env_trigonometric_rational(), iterations)
    print("example sequence functionality: {:05.3f}".format(f))

    _artificial_isolated(iterations)
    _artificial_transfer(iterations)

    pyplot.legend()
    pyplot.show()


def natural_experiment(iterations: int = 500000):
    c = Config("../configs/config.json")

    env = env_crypto(c["data_dir"] + "23Jun2017-23Jun2018-1m/EOSETH.csv", 60, start_val=1501113780, end_val=1529712000)
    f = prediction_functionality(env, iterations, rational=True)
    print("sequence a functionality: {:05.3f}".format(f))
    env = env_crypto(c["data_dir"] + "23Jun2017-23Jun2018-1m/QTUMETH.csv", 60, start_val=1501113780, end_val=1529712000)
    f = prediction_functionality(env, iterations, rational=True)
    print("sequence b functionality: {:05.3f}".format(f))

    _natural_isolated(iterations)
    _natural_transfer(iterations)

    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    natural_experiment()
    # artificial_experiment()

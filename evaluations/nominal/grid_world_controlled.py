# coding=utf-8
from matplotlib import pyplot

from data.controllers import sarsa_nominal_controller
from environments.interactive import env_grid_world
from evaluations.experiments import interaction
from modelling.predictors.nominal.semiotic import NominalSemioticModel
from tools.load_configs import Config


def main():
    rotational = True
    iterations = 100000

    c = Config("../../configs/config.json")
    if rotational:
        movement = "f", "b", "r", "l"
    else:
        movement = "n", "e", "s", "w"
    controller = sarsa_nominal_controller(movement, alpha=.1, gamma=.1, epsilon=.1)
    environment = env_grid_world(c["data_dir"] + "grid_worlds/snake.txt")

    predictor = NominalSemioticModel(no_examples=1, alpha=50, sigma=.1, trace_length=1)

    interaction(environment, controller, predictor, rational=False, iterations=iterations)

    pyplot.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# coding=utf-8
import os

from matplotlib import pyplot

from data_generation.deprecated_controllers import random_nominal_controller
from data_generation.deprecated_example_generation import example_interactive, example_interactive_senses, rationalize_generator
from environments.deprecated_interactive import env_grid_world
from setups_deprecated.evaluations import prediction
from modelling.predictors.rational.baseline import Regression
from modelling.predictors.rational.semiotic import RationalSemioticModel
from tools.load_configs import Config


def _isolated(iterations: int, rotational: bool):
    c = Config("../../configs/config.json")
    if rotational:
        movement = "f", "b", "r", "l"
    else:
        movement = "n", "e", "s", "w"
    controller = random_nominal_controller(movement)
    examples = example_interactive(env_grid_world(c["data_dir"] + "grid_worlds/snake.txt"), controller, history_length=1)

    predictor_a = Regression(input_dimension=5, output_dimension=4, no_examples=1, drag=100)
    predictor_b = RationalSemioticModel(input_dimension=5, output_dimension=4, no_examples=1, alpha=10, sigma=.5, drag=100, history_length=1)
    predictors = predictor_a, predictor_b

    prediction(rationalize_generator(examples), predictors, iterations=iterations, rational=True)


def _analysis(iterations: int, rotational: bool):
    c = Config("../../configs/config.json")
    if rotational:
        movement = "f", "b", "r", "l"
    else:
        movement = "n", "e", "s", "w"
    controller = random_nominal_controller(movement)
    examples = example_interactive_senses(env_grid_world(c["data_dir"] + "grid_worlds/snake.txt"), controller, history_length=1)

    predictor_a = Regression(input_dimension=2, output_dimension=1, no_examples=4, drag=100)
    predictor_b = RationalSemioticModel(input_dimension=2, output_dimension=1, no_examples=4, alpha=10, sigma=.5, drag=100, history_length=1)
    predictors = predictor_a, predictor_b

    prediction(rationalize_generator(examples), predictors, iterations=iterations, rational=True)


if __name__ == "__main__":
    duration = 100000
    rotation = True

    pyplot.suptitle(os.path.basename(__file__))

    _isolated(duration, rotation)
    _analysis(duration, rotation)

    pyplot.show()

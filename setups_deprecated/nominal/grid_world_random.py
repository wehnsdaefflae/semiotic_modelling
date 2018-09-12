#!/usr/bin/env python3
# coding=utf-8
import os

from matplotlib import pyplot

from data_generation.deprecated_controllers import random_nominal_controller
from data_generation.deprecated_example_generation import example_interactive, example_interactive_senses
from environments.deprecated_interactive import env_grid_world
from setups_deprecated.evaluations import prediction
from modelling.predictors.nominal.semiotic import NominalSemioticModel
from modelling.predictors.nominal.baseline import NominalMarkovModel
from tools.load_configs import Config


def _isolated(iterations: int, rotational: bool):
    c = Config("../../configs/config.json")
    if rotational:
        movement = "f", "b", "r", "l"
    else:
        movement = "n", "e", "s", "w"
    controller = random_nominal_controller(movement)
    examples = example_interactive(env_grid_world(c["data_dir"] + "grid_worlds/simple.txt"), controller, history_length=1)

    predictor_a = NominalMarkovModel(no_examples=1)
    predictor_b = NominalSemioticModel(no_examples=1, alpha=50, sigma=.2, trace_length=1)
    predictors = predictor_a, predictor_b

    prediction(examples, predictors, iterations=iterations, rational=False)


def _analysis(iterations: int, rotational: bool):
    c = Config("../../configs/config.json")
    if rotational:
        movement = "f", "b", "r", "l"
    else:
        movement = "n", "e", "s", "w"
    controller = random_nominal_controller(movement)
    examples = example_interactive_senses(env_grid_world(c["data_dir"] + "grid_worlds/simple.txt"), controller, history_length=1)

    predictor_a = NominalMarkovModel(no_examples=4)
    predictor_b = NominalSemioticModel(no_examples=4, alpha=50, sigma=.2, trace_length=1)
    predictors = predictor_a, predictor_b

    prediction(examples, predictors, iterations=iterations, rational=False)


if __name__ == "__main__":
    duration = 200000
    rotation = True

    pyplot.suptitle(os.path.basename(__file__))

    _isolated(duration, rotation)
    _analysis(duration, rotation)

    pyplot.show()
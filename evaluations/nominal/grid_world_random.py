#!/usr/bin/env python3
# coding=utf-8
import os

from matplotlib import pyplot

from data.controllers import random_nominal_controller
from data.example_generation import example_interactive, example_interactive_senses
from environments.interactive import env_grid_world
from evaluations.experiments import prediction
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
    examples = example_interactive(env_grid_world(c["data_dir"] + "grid_worlds/square.txt"), controller, history_length=1)

    predictor_a = NominalMarkovModel(no_examples=1)
    predictor_b = NominalSemioticModel(no_examples=1, alpha=100, sigma=.2, trace_length=1)
    predictors = predictor_a, predictor_b

    prediction(examples, predictors, iterations=iterations, rational=False)


def _analysis(iterations: int, rotational: bool):
    c = Config("../../configs/config.json")
    if rotational:
        movement = "f", "b", "r", "l"
    else:
        movement = "n", "e", "s", "w"
    controller = random_nominal_controller(movement)
    examples = example_interactive_senses(env_grid_world(c["data_dir"] + "grid_worlds/square.txt"), controller, history_length=1)

    predictor_a = NominalMarkovModel(no_examples=4)
    predictor_b = NominalSemioticModel(no_examples=4, alpha=100, sigma=.2, trace_length=1)
    predictors = predictor_a, predictor_b

    prediction(examples, predictors, iterations=iterations, rational=False)


if __name__ == "__main__":
    duration = 500000
    rotation = False

    pyplot.suptitle(os.path.basename(__file__))

    _isolated(duration, rotation)
    _analysis(duration, rotation)

    pyplot.show()

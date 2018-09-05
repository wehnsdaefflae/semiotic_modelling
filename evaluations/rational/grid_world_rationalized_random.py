#!/usr/bin/env python3
# coding=utf-8

from matplotlib import pyplot

from data.example_generation import example_random_interactive, example_random_interactive_senses, rationalize_generator
from environments.interactive import env_grid_world
from evaluations.experiments import experiment
from modelling.predictors.rational.baseline import Regression
from modelling.predictors.rational.semiotic import RationalSemioticModel
from tools.load_configs import Config


def _isolated(iterations: int, rotational: bool):
    c = Config("../../configs/config.json")
    if rotational:
        movement = "f", "b", "r", "l"
    else:
        movement = "n", "e", "s", "w"
    examples = example_random_interactive(env_grid_world(c["data_dir"] + "grid_worlds/snake.txt"), movement, history_length=1)

    predictor_a = Regression(input_dimension=5, output_dimension=4, no_examples=1, drag=100)
    predictor_b = RationalSemioticModel(input_dimension=5, output_dimension=4, no_examples=1, alpha=10, sigma=.5, drag=100, trace_length=1)
    predictors = predictor_a, predictor_b

    experiment(rationalize_generator(examples), predictors, iterations=iterations, rational=False)


def _analysis(iterations: int, rotational: bool):
    c = Config("../../configs/config.json")
    if rotational:
        movement = "f", "b", "r", "l"
    else:
        movement = "n", "e", "s", "w"
    examples = example_random_interactive_senses(env_grid_world(c["data_dir"] + "grid_worlds/snake.txt"), movement, history_length=1)

    predictor_a = Regression(input_dimension=2, output_dimension=1, no_examples=4, drag=100)
    predictor_b = RationalSemioticModel(input_dimension=2, output_dimension=1, no_examples=4, alpha=10, sigma=.5, drag=100, trace_length=1)
    predictors = predictor_a, predictor_b

    experiment(rationalize_generator(examples), predictors, iterations=iterations, rational=False)


if __name__ == "__main__":
    duration = 100000
    rotation = True

    _isolated(duration, rotation)
    _analysis(duration, rotation)

    pyplot.tight_layout()
    pyplot.show()

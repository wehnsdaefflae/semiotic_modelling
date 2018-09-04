#!/usr/bin/env python3
# coding=utf-8

from matplotlib import pyplot

from data.example_generation import example_random_interactive, example_random_interactive_senses
from environments.interactive import env_grid_world
from evaluations.experiments import experiment
from modelling.predictors.nominal.semiotic import NominalSemioticModel
from modelling.predictors.nominal.baseline import NominalMarkovModelIsolated, NominalMarkovModelIntegrated
from tools.load_configs import Config


def _isolated(iterations: int, rotational: bool):
    c = Config("../configs/config.json")
    if rotational:
        movement = "f", "b", "r", "l"
    else:
        movement = "n", "e", "s", "w"
    examples = example_random_interactive(env_grid_world(c["data_dir"] + "grid_worlds/snake.txt"), movement, history_length=1)

    predictor_a = NominalMarkovModelIntegrated(no_examples=1)
    predictor_b = NominalSemioticModel(no_examples=1, alpha=50, sigma=.1, trace_length=1)
    predictors = predictor_a, predictor_b

    experiment(examples, predictors, iterations=iterations, rational=False)


def _analysis(iterations: int, rotational: bool):
    c = Config("../configs/config.json")
    if rotational:
        movement = "f", "b", "r", "l"
    else:
        movement = "n", "e", "s", "w"
    examples = example_random_interactive_senses(env_grid_world(c["data_dir"] + "grid_worlds/snake.txt"), movement, history_length=1)

    predictor_a = NominalMarkovModelIsolated(no_examples=4)
    predictor_b = NominalSemioticModel(no_examples=4, alpha=50, sigma=.1, trace_length=1)
    predictors = predictor_a, predictor_b

    experiment(examples, predictors, iterations=iterations, rational=False)


def _experiment(iterations: int = 100000, rotational: bool = False):
    _isolated(iterations, rotational)
    _analysis(iterations, rotational)

    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    _experiment(rotational=True)

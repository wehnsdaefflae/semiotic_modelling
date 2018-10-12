# coding=utf-8
from typing import Tuple, Collection, Iterable, Any, Iterator, _T_co

from data_generation.data_sources.systems.abstract_classes import Environment, Controller
from modelling.predictors.abstract_predictor import Predictor
from visualization.visualization import Visualize

EXPERIMENT = Tuple[Environment, Controller, Predictor]


class SetupInteraction(Iterable[Tuple[Any, ...]]):

    def __iter__(self) -> Iterator[_T_co]:
        pass


def setup(experiments: Collection[EXPERIMENT], rational: bool, iterations: int = 500000, repetitions: int = 20):
    labels = tuple(f"{_e.__class__.__name__}, {_c.__class__.__name__}, {_p.__class__.__name__}" for _e, _c, _p in experiments)
    plots = {"reward": labels, "error": labels, "duration": labels}
    Visualize.init("interactive experiment", plots, x_range=iterations, refresh_rate=iterations // 1000)

    for repetition in range(repetitions):
        for iteration in range(iterations):
            for environment, controller, predictor in experiments:
                pass

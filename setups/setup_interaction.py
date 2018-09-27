# coding=utf-8
from data_generation.data_sources.systems.abstract_classes import Environment, Controller
from modelling.predictors.abstract_predictor import Predictor


def setup(predictor: Predictor, environment: Environment, controller: Controller, rational: bool, iterations: int = 500000, repetitions: int = 20):
    pass

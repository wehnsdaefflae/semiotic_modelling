# coding=utf-8
from typing import Any, Type, TypeVar

from _framework.systems_abstract import Controller, Predictor

MOTOR_TYPE = TypeVar("MOTOR_TYPE")
SENSOR_TYPE = TypeVar("SENSOR_TYPE")


class NoneController(Controller[SENSOR_TYPE, Type[None]]):
    def __init__(self):
        super().__init__()

    def _integrate(self, evaluation: float):
        return

    def _react(self, data_in: Any) -> Type[None]:
        return None


class SarsaController(Controller[SENSOR_TYPE, MOTOR_TYPE]):
    def __init__(self, predictor: Predictor[MOTOR_TYPE, SENSOR_TYPE] = None):
        self.predictor = predictor      # if predictor present attach state to sensor data
        raise NotImplementedError()

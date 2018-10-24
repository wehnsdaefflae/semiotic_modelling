# coding=utf-8
from typing import TypeVar, Generic, Tuple, Optional, Sequence, Collection

from tools.io_tools import PersistenceMixin

INPUT_TYPE = TypeVar("INPUT_TYPE")
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")
EXAMPLE = Tuple[INPUT_TYPE, OUTPUT_TYPE]


class System(Generic[INPUT_TYPE, OUTPUT_TYPE], PersistenceMixin):
    def __str__(self):
        return self.__class__.__name__

    def _react(self, data_in: Sequence[INPUT_TYPE]) -> Sequence[OUTPUT_TYPE]:
        raise NotImplementedError()

    def react(self, data_in: Sequence[INPUT_TYPE]) -> Sequence[OUTPUT_TYPE]:
        return self._react(data_in)


class Predictor(System[INPUT_TYPE, OUTPUT_TYPE]):
    def _react(self, data_in: Sequence[INPUT_TYPE]) -> Sequence[OUTPUT_TYPE]:
        raise NotImplementedError()

    def fit(self, data_in: Sequence[INPUT_TYPE], data_out: Sequence[OUTPUT_TYPE]):
        raise NotImplementedError()

    def predict(self, data_in: Sequence[INPUT_TYPE]) -> Sequence[OUTPUT_TYPE]:
        return self._react(data_in)


SENSOR_TYPE = TypeVar("SENSOR_TYPE")
MOTOR_TYPE = TypeVar("MOTOR_TYPE")


class Task(System[MOTOR_TYPE, SENSOR_TYPE]):
    def _react(self, data_in: MOTOR_TYPE) -> SENSOR_TYPE:
        raise NotImplementedError()

    def _evaluate_action(self, data_in: MOTOR_TYPE) -> float:
        raise NotImplementedError()

    def motor_space(self) -> Collection[MOTOR_TYPE]:
        raise NotImplementedError

    def respond(self, data_in: Optional[MOTOR_TYPE]) -> Tuple[SENSOR_TYPE, float]:
        return self._react(data_in), self._evaluate_action(data_in)


class Controller(System[SENSOR_TYPE, MOTOR_TYPE]):
    def _react(self, data_in: SENSOR_TYPE) -> MOTOR_TYPE:
        raise NotImplementedError()

    def _integrate(self, evaluation: float):
        raise NotImplementedError()

    def decide(self, data_in: Optional[SENSOR_TYPE], evaluation: float) -> MOTOR_TYPE:
        self._integrate(evaluation)
        return self._react(data_in)

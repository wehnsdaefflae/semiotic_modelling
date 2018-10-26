# coding=utf-8
from typing import TypeVar, Generic, Tuple, Optional, Collection

from tools.io_tools import PersistenceMixin

INPUT_TYPE = TypeVar("INPUT_TYPE")
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")


class System(Generic[INPUT_TYPE, OUTPUT_TYPE], PersistenceMixin):
    def __str__(self):
        return self.__class__.__name__

    def _react(self, data_in: Tuple[INPUT_TYPE]) -> Tuple[OUTPUT_TYPE]:
        raise NotImplementedError()

    def react(self, data_in: Tuple[INPUT_TYPE]) -> Tuple[OUTPUT_TYPE]:
        return self._react(data_in)

    def get_state(self) -> Tuple[Tuple[int, ...], ...]:
        raise NotImplementedError()


class Predictor(System[INPUT_TYPE, OUTPUT_TYPE]):
    def _react(self, data_in: Tuple[INPUT_TYPE]) -> Tuple[OUTPUT_TYPE]:
        raise NotImplementedError()

    def fit(self, data_in: Tuple[INPUT_TYPE], data_out: Tuple[OUTPUT_TYPE]):
        raise NotImplementedError()

    def predict(self, data_in: Tuple[INPUT_TYPE]) -> Tuple[OUTPUT_TYPE]:
        return self._react(data_in)

    def get_state(self) -> Tuple[Tuple[int, ...], ...]:
        raise NotImplementedError()


SENSOR_TYPE = TypeVar("SENSOR_TYPE")
MOTOR_TYPE = TypeVar("MOTOR_TYPE")


class Task(System[MOTOR_TYPE, SENSOR_TYPE]):
    def _react(self, data_in: MOTOR_TYPE) -> SENSOR_TYPE:
        raise NotImplementedError()

    def _evaluate_action(self, data_in: MOTOR_TYPE) -> float:
        raise NotImplementedError()

    @staticmethod
    def motor_space() -> Collection[MOTOR_TYPE]:
        raise NotImplementedError

    def respond(self, data_in: Optional[MOTOR_TYPE]) -> Tuple[SENSOR_TYPE, float]:
        return self._react(data_in), self._evaluate_action(data_in)

    def get_state(self) -> Tuple[Tuple[int, ...], ...]:
        raise NotImplementedError()


class Controller(System[SENSOR_TYPE, MOTOR_TYPE]):
    def _react(self, data_in: SENSOR_TYPE) -> MOTOR_TYPE:
        raise NotImplementedError()

    def integrate(self, data_in: Optional[SENSOR_TYPE], evaluation: float):
        raise NotImplementedError()

    def decide(self, data_in: Optional[SENSOR_TYPE]) -> MOTOR_TYPE:
        return self._react(data_in)

    def get_state(self) -> Tuple[Tuple[int, ...], ...]:
        raise NotImplementedError()

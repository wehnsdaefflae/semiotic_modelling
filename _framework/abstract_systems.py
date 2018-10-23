# coding=utf-8
from typing import TypeVar, Generic, Tuple

INPUT_TYPE = TypeVar("INPUT_TYPE")
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")


class System(Generic[INPUT_TYPE, OUTPUT_TYPE]):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def _react(self, data_in: INPUT_TYPE) -> OUTPUT_TYPE:
        raise NotImplementedError()

    def react(self, data_in: INPUT_TYPE) -> OUTPUT_TYPE:
        return self._react(data_in)

    @staticmethod
    def error(data_a: OUTPUT_TYPE, data_b: OUTPUT_TYPE) -> float:
        return float(data_a != data_b)


class Predictor(System[INPUT_TYPE, OUTPUT_TYPE]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _react(self, data_in: INPUT_TYPE) -> OUTPUT_TYPE:
        raise NotImplementedError()

    def predict(self, data_in: INPUT_TYPE) -> OUTPUT_TYPE:
        return self._react(data_in)


class Task(System[INPUT_TYPE, OUTPUT_TYPE]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _react(self, data_in: INPUT_TYPE) -> OUTPUT_TYPE:
        raise NotImplementedError()

    def _evaluate(self, data_in: INPUT_TYPE) -> float:
        raise NotImplementedError()

    def respond(self, data_in: INPUT_TYPE) -> Tuple[OUTPUT_TYPE, float]:
        return self._react(data_in), self._evaluate(data_in)


class Controller(System[INPUT_TYPE, OUTPUT_TYPE]):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def _react(self, data_in: INPUT_TYPE) -> OUTPUT_TYPE:
        raise NotImplementedError()

    def _integrate(self, evaluation: float):
        raise NotImplementedError()

    def decide(self, data_in: INPUT_TYPE, eval_in: float) -> OUTPUT_TYPE:
        self._integrate(eval_in)
        return self._react(data_in)

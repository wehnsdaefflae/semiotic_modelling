# coding=utf-8
from typing import Tuple

from _framework.data_types import NOMINAL_INPUT, NOMINAL_OUTPUT
from _framework.streams.abstract import ExampleStream


class NominalStream(ExampleStream[NOMINAL_INPUT, NOMINAL_OUTPUT]):
    def __init__(self, learn_control: bool, *args, **kwargs):
        super().__init__(learn_control, *args, **kwargs)

    def __str__(self):
        raise NotImplementedError()

    def _before(self):
        raise NotImplementedError()

    def _get_inputs(self) -> Tuple[NOMINAL_INPUT, ...]:
        raise NotImplementedError()

    def _get_outputs(self) -> Tuple[NOMINAL_OUTPUT, ...]:
        raise NotImplementedError()

    def _after(self):
        raise NotImplementedError()

    def _single_error(self, data_output: NOMINAL_OUTPUT, data_target: NOMINAL_OUTPUT) -> float:
        return float(data_output != data_target)

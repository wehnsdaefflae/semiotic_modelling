# coding=utf-8
from math import sqrt
from typing import Tuple

from _framework.data_types import RATIONAL_INPUT, RATIONAL_OUTPUT
from _framework.streams.abstract import ExampleStream


class RationalStream(ExampleStream[RATIONAL_INPUT, RATIONAL_OUTPUT]):
    def __init__(self, learn_control: bool, *args, **kwargs):
        super().__init__(learn_control, *args, **kwargs)

    def __str__(self):
        raise NotImplementedError()

    def _before(self):
        raise NotImplementedError()

    def _get_inputs(self) -> Tuple[RATIONAL_INPUT, ...]:
        raise NotImplementedError()

    def _get_outputs(self) -> Tuple[RATIONAL_OUTPUT, ...]:
        raise NotImplementedError()

    def _after(self):
        raise NotImplementedError()

    def _single_error(self, data_output: RATIONAL_OUTPUT, data_target: RATIONAL_OUTPUT) -> float:
        return sqrt(sum((_o - _t) ** 2. for _o, _t in zip(data_output, data_target)))

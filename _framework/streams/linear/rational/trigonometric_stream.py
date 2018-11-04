# coding=utf-8
from math import sin, cos

from _framework.data_types import RATIONAL_INPUT, RATIONAL_OUTPUT
from _framework.streams.linear.rational.abstract import RationalStream


class TrigonometricStream(RationalStream[RATIONAL_INPUT, RATIONAL_OUTPUT]):
    def __init__(self):
        super().__init__(False)
        self._iteration = 0

    def __str__(self):
        return self.__class__.__name__

    def _before(self):
        pass

    def _get_inputs(self) -> RATIONAL_INPUT:
        input_value = sin(self._iteration / 100.),
        return input_value,

    def _get_outputs(self) -> RATIONAL_OUTPUT:
        output_value = float(cos(self._iteration / 100.) >= 0.) * 2. - 1.,
        return output_value,

    def _after(self):
        self._iteration += 1

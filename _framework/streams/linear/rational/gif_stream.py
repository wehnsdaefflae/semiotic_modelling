# coding=utf-8
from typing import Tuple

from _framework.data_types import RATIONAL_INPUT, RATIONAL_OUTPUT
from _framework.streams.linear.rational.abstract import RationalStream


# TODO: finish gif stream
class GifStream(RationalStream[RATIONAL_INPUT, RATIONAL_OUTPUT]):
    def __init__(self):
        super().__init__(False)

    def __str__(self):
        pass

    def _before(self):
        pass

    def _get_inputs(self) -> Tuple[RATIONAL_INPUT, ...]:
        pass

    def _get_outputs(self) -> Tuple[RATIONAL_OUTPUT, ...]:
        pass

    def _after(self):
        pass

    @staticmethod
    def _single_error(data_output: RATIONAL_OUTPUT, data_target: RATIONAL_OUTPUT) -> float:
        pass

# coding=utf-8
from typing import Tuple

from _framework.data_types import RATIONAL_INPUT, RATIONAL_OUTPUT
from _framework.streams.linear.rational.abstract import RationalStream


# TODO: finish gif stream
class GifStream(RationalStream[RATIONAL_INPUT, RATIONAL_OUTPUT]):
    def __init__(self, *args, **kwargs):
        super().__init__(False, *args, **kwargs)

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

    def _single_error(self, data_output: RATIONAL_OUTPUT, data_target: RATIONAL_OUTPUT) -> float:
        pass

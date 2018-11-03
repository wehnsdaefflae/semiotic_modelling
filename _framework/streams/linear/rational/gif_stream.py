# coding=utf-8
from typing import Tuple

from _framework.data_types import RATIONAL_INPUT, RATIONAL_OUTPUT
from _framework.streams.abstract import ExampleStream, OUTPUT_TYPE, INPUT_TYPE


class GifStream(ExampleStream[RATIONAL_INPUT, RATIONAL_OUTPUT]):
    def __init__(self):
        super().__init__(False)

    def __str__(self):
        pass

    def _before(self):
        pass

    def _get_inputs(self) -> Tuple[INPUT_TYPE, ...]:
        pass

    def _get_outputs(self) -> Tuple[OUTPUT_TYPE, ...]:
        pass

    def _after(self):
        pass

# coding=utf-8
from typing import Tuple

from _framework.data_types import RATIONAL_OUTPUT, RATIONAL_INPUT
from _framework.streams.linear.nominal.abstract import NominalStream
from _framework.streams.linear.rational.abstract import RationalStream


# TODO: implement rationalizer
class RationalizedStream(RationalStream):
    def __init__(self, nominal_stream: NominalStream, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nominal_stream = nominal_stream

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
# coding=utf-8
import random
import string
from typing import Tuple

from _framework.data_types import NOMINAL_INPUT, NOMINAL_OUTPUT
from _framework.streams.abstract import OUTPUT_TYPE, INPUT_TYPE
from _framework.streams.linear.nominal.abstract import NominalStream

NOMINAL_HISTORY = Tuple[NOMINAL_INPUT, ...]
NOMINAL_EXAMPLE = Tuple[NOMINAL_HISTORY, NOMINAL_OUTPUT]


class NominalAscendingDescending(NominalStream[NOMINAL_INPUT, NOMINAL_OUTPUT]):
    def __init__(self, *args, **kwargs):
        super().__init__(False, *args, **kwargs)
        self._index = 0
        self._state = False

        self._sequence = string.digits
        self._len = len(self._sequence)

    def __str__(self):
        return self.__class__.__name__

    def _before(self):
        pass

    def _get_inputs(self) -> Tuple[INPUT_TYPE, ...]:
        return self._sequence[self._index],

    def _get_outputs(self) -> Tuple[OUTPUT_TYPE, ...]:
        neighbor_index = int((self._index + float(self._state) * 2. - 1.) % self._len)
        return self._sequence[neighbor_index],

    def _after(self):
        if random.random() < .2:
            self._state = not self._state

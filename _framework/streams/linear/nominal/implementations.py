# coding=utf-8
import random
import string
from typing import Tuple

from _framework.data_types import NOMINAL_INPUT, NOMINAL_OUTPUT
from _framework.streams.abstract import ExampleStream

NOMINAL_HISTORY = Tuple[NOMINAL_INPUT, ...]
NOMINAL_EXAMPLE = Tuple[NOMINAL_HISTORY, NOMINAL_OUTPUT]


class NominalAscendingDescending(ExampleStream[NOMINAL_INPUT, NOMINAL_OUTPUT]):
    def __init__(self, history_length: int = 1):
        super().__init__(False, history_length=history_length)
        self._index = 0
        self._state = False

        self._sequence = string.digits
        self._len = len(self._sequence)

    def __str__(self):
        return self.__class__.__name__

    def next(self) -> Tuple[NOMINAL_EXAMPLE, ...]:
        input_data = self._sequence[self._index]
        self._history.append(input_data)

        self._index = int((self._index + float(self._state) * 2. - 1.) % self._len)
        output_data = self._sequence[self._index]
        if random.random() < .2:
            self._state = not self._state
        return (tuple(self._history), output_data),

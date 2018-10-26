# coding=utf-8
import string
import random
from typing import Tuple

from _framework.streams_abstract import ExampleStream, EXAMPLE


class NominalAscendingDescending(ExampleStream[str, str]):
    def __init__(self):
        super().__init__(False)
        self._index = 0
        self._state = False

        self._sequence = string.digits
        self._len = len(self._sequence)

    def __str__(self):
        return self.__class__.__name__

    def next(self) -> Tuple[EXAMPLE, ...]:
        input_data = self._sequence[self._index]
        self._index = int((self._index + float(self._state) * 2. - 1.) % self._len)
        output_data = self._sequence[self._index]
        if random.random() < .2:
            self._state = not self._state
        return (input_data, output_data),

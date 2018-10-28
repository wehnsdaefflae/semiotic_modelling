# coding=utf-8
from typing import TypeVar, Generic, Tuple

from tools.io_tools import PersistenceMixin

INPUT_TYPE = TypeVar("INPUT_TYPE")
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")


class System(PersistenceMixin, Generic[INPUT_TYPE, OUTPUT_TYPE]):
    def __str__(self):
        return self.__class__.__name__

    def _react(self, data_in: Tuple[INPUT_TYPE]) -> Tuple[OUTPUT_TYPE]:
        raise NotImplementedError()

    def react(self, data_in: Tuple[INPUT_TYPE]) -> Tuple[OUTPUT_TYPE]:
        return self._react(data_in)

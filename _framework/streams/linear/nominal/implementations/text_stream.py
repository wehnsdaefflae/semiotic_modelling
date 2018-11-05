# coding=utf-8
from typing import Tuple

from _framework.data_types import NOMINAL_INPUT, NOMINAL_OUTPUT
from _framework.streams.linear.nominal.abstract import NominalStream
from _framework.streams.linear.nominal.implementations.resources.text_generator import sequence_nominal_text


class TextStream(NominalStream):
    def __init__(self, file_path: str, *args, **kwargs):
        super().__init__(False, no_examples=1, *args, **kwargs)
        self._file_path = file_path
        self._input_sequence = sequence_nominal_text(self._file_path)
        self._target_sequence = sequence_nominal_text(self._file_path)
        next(self._target_sequence)

    def __str__(self):
        return self._file_path

    def _before(self):
        pass

    def _get_inputs(self) -> Tuple[NOMINAL_INPUT, ...]:
        return next(self._input_sequence),

    def _get_outputs(self) -> Tuple[NOMINAL_OUTPUT, ...]:
        return next(self._target_sequence),

    def _after(self):
        pass

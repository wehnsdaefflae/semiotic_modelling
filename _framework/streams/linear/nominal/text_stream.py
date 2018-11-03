# coding=utf-8
from typing import Tuple

from _framework.streams.abstract import ExampleStream, OUTPUT_TYPE, INPUT_TYPE
from _framework.streams.linear.nominal.resources.text_generator import sequence_nominal_text


class TextStream(ExampleStream[INPUT_TYPE, OUTPUT_TYPE]):
    def __init__(self, file_path: str, history_length: int = 1):
        super().__init__(False, no_examples=1, history_length=history_length)
        self._file_path = file_path
        self._input_sequence = sequence_nominal_text(self._file_path)
        self._target_sequence = sequence_nominal_text(self._file_path)
        next(self._target_sequence)

    def __str__(self):
        return self._file_path

    def _before(self):
        pass

    def _get_inputs(self) -> Tuple[INPUT_TYPE, ...]:
        return next(self._input_sequence),

    def _get_outputs(self) -> Tuple[OUTPUT_TYPE, ...]:
        return next(self._target_sequence),

    def _after(self):
        pass

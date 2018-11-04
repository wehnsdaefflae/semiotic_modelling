# coding=utf-8
from typing import Tuple, Sequence, Optional, Union

from _framework.data_types import RATIONAL_INPUT, RATIONAL_OUTPUT
from _framework.streams.linear.nominal.resources.crypto_generator import sequence_rational_crypto
from _framework.streams.linear.rational.abstract import RationalStream


# TODO: finish implementing exchange rate stream
class ExchangeRateStream(RationalStream[RATIONAL_INPUT, RATIONAL_OUTPUT]):
    def __init__(self,
                 input_file_paths: Sequence[str], target_file_paths: Sequence[str],
                 interval_seconds: int, offset_seconds: int,
                 start_val: Optional[Union[int, str]] = None,
                 end_val: Optional[Union[int, str]]= None):

        super().__init__(False)
        self._input_sequences = tuple(sequence_rational_crypto(each_path, interval_seconds, start_val, end_val) for each_path in input_file_paths)
        self._target_sequences = tuple(sequence_rational_crypto(each_path, interval_seconds, start_val + offset_seconds, end_val + offset_seconds) for each_path in target_file_paths)

    def __str__(self):
        pass

    def _before(self):
        pass

    def _get_inputs(self) -> Tuple[RATIONAL_INPUT, ...]:
        return tuple(next(each_sequence) for each_sequence in self._input_sequences)

    def _get_outputs(self) -> Tuple[RATIONAL_OUTPUT, ...]:
        return tuple(next(each_sequence) for each_sequence in self._target_sequences)

    def _after(self):
        pass

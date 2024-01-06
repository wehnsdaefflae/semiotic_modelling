# coding=utf-8
from typing import Tuple, Sequence

from _framework.data_types import RATIONAL_INPUT, RATIONAL_OUTPUT
from _framework.streams.linear.rational.implementations.resources.crypto_generator import equisample, binance_generator
from _framework.streams.linear.rational.abstract import RationalStream
from dateutil import parser


class ExchangeRateStream(RationalStream):
    def __init__(self,
                 input_file_paths: Sequence[str], target_file_paths: Sequence[str],
                 start_time: str, end_time: str,
                 interval_seconds: int, offset_seconds: int,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        dt_obj = parser.parse(start_time)
        start_ts = dt_obj.timestamp()

        dt_obj = parser.parse(end_time)
        end_ts = dt_obj.timestamp()

        self._input_sequences = tuple(
            equisample(binance_generator(_each_path, start_timestamp=start_ts, end_timestamp=end_ts - offset_seconds), interval_seconds)
            for _each_path in input_file_paths
        )

        self._target_sequences = tuple(
            equisample(binance_generator(_each_path, start_timestamp=start_ts + offset_seconds, end_timestamp=end_ts), interval_seconds)
            for _each_path in target_file_paths
        )

        self._input_data = None

    def __str__(self):
        return self.__class__.__name__

    def _before(self):
        self._input_data = tuple(next(each_sequence) for each_sequence in self._input_sequences)

    def _get_inputs(self) -> Tuple[RATIONAL_INPUT, ...]:
        return self._input_data,

    def _get_outputs(self) -> Tuple[RATIONAL_OUTPUT, ...]:
        future_data = tuple(next(each_sequence) for each_sequence in self._target_sequences)
        target_data = tuple(float(1. - _n / _t >= .05) for _n, _t in zip(future_data, self._input_data))
        return target_data,

    def _after(self):
        pass

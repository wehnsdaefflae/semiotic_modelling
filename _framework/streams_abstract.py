# coding=utf-8
import collections
import numbers
from math import sqrt
from typing import TypeVar, Generic, Tuple, Sequence

INPUT_TYPE = TypeVar("INPUT_TYPE")
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE")


class ExampleStream(Generic[INPUT_TYPE, OUTPUT_TYPE]):
    def __init__(self, *args, **kwargs):
        self._last_reward = 0.

    def __str__(self):
        raise NotImplementedError()

    def next(self) -> Tuple[Tuple[INPUT_TYPE, OUTPUT_TYPE], ...]:
        raise NotImplementedError()

    def get_last_reward(self) -> float:
        return self._last_reward

    @staticmethod
    def error(data_output: Sequence[OUTPUT_TYPE], data_target: Sequence[OUTPUT_TYPE]) -> float:
        d = len(data_output)
        assert d == len(data_target)

        error_sum = 0.
        for _o, _t in zip(data_output, data_target):
            assert type(_o) == type(_t)

            # vectors
            if isinstance(_o, collections.Sequence):
                sub_sum = 0.
                for __o, __t in zip(_o, _t):
                    sub_sum += (__o - __t) ** 2.
                error_sum += sqrt(sub_sum)

            # scalars
            elif isinstance(_o, numbers.Rational):
                error_sum += _o - _t

            # anything else
            else:
                error_sum += float(_o != _t)

        return error_sum / d

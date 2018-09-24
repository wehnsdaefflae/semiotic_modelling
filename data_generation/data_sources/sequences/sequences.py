#!/usr/bin/env python3
# coding=utf-8
import string
from math import sin, cos
from typing import Optional, Union, Iterator

from dateutil import parser

from data_generation.data_processing import series_generator, equisample
from data_generation.data_sources.sequences.non_interactive import sequence_nominal_alternating


class ArtificialNominal(Iterator[str]):
    def __init__(self):
        self.generator = sequence_nominal_alternating()

    def __next__(self) -> str:
        return next(self.generator)


class Text(Iterator[str]):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.permissible_non_letter = string.digits + string.punctuation + " "
        with open(self.file_path, mode="r") as file:
            self.text = file.read()
        self.text_len = len(self.text)
        self.index = -1

    def __next__(self) -> str:
        while True:
            self.index = (self.index + 1) % self.text_len
            element = self.text[self.index]

            if element in string.ascii_letters:
                return element.lower()

            if element in self.permissible_non_letter:
                return element


class ArtificialRationalCosine(Iterator[float]):
    def __init__(self):
        self.i = -1

    def __next__(self) -> float:
        return float(cos(self.i / 100.) >= 0.) * 2. - 1.


class ArtificialRationalSinus(Iterator[float]):
    def __init__(self):
        self.i = -1

    def __next__(self) -> float:
        self.i += 1
        return sin(self.i / 100.)


class ExchangeRates(Iterator[float]):
    def __init__(self, file_path: str, interval_seconds: int,
                 start_val: Optional[Union[int, str]] = None, end_val: Optional[Union[int, str]] = None):
        self.start_ts = ExchangeRates._convert_to_timestamp(start_val)
        self.end_ts = ExchangeRates._convert_to_timestamp(end_val)

        self.raw_generator = series_generator(file_path, start_timestamp=self.start_ts, end_timestamp=self.end_ts)
        self.generator = equisample(self.raw_generator, interval_seconds)

        self.file_path = file_path
        self.interval_seconds = interval_seconds

    @staticmethod
    def _convert_to_timestamp(time_val: Optional[Union[int, str]]) -> int:
        if time_val is None:
            return -1

        time_type = type(time_val)
        if time_type == int:
            return time_val

        elif time_type == str:
            date_time = parser.parse(time_val)
            return date_time.timestamp()

        raise ValueError()

    def __next__(self) -> float:
        try:
            t, value = next(self.generator)

        except StopIteration:
            self.raw_generator = series_generator(self.file_path, start_timestamp=self.start_ts, end_timestamp=self.end_ts)
            self.generator = equisample(self.raw_generator, self.interval_seconds)
            t, value = next(self.generator)

        return value

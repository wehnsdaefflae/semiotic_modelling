#!/usr/bin/env python3
# coding=utf-8
import string
from typing import Optional, TypeVar

from data.systems.abstract_classes import Environment, EXPERIENCE
from environments.non_interactive import sequence_rational_crypto, sequence_nominal_alternating

SEQUENCE_ELEMENT = TypeVar("SEQUENCE_ELEMENT")
NO_ACTION = None


# TODO: don't return reward just to integrate interaction and noninteraction. change experiment instead


class Sequence(Environment[SEQUENCE_ELEMENT, NO_ACTION]):
    def react_to(self, motor: Optional[NO_ACTION]) -> EXPERIENCE[SEQUENCE_ELEMENT]:
        raise NotImplementedError()


class ArtificialNominal(Sequence[str]):
    def __init__(self):
        self.generator = sequence_nominal_alternating()

    def react_to(self, motor: Optional[NO_ACTION]) -> EXPERIENCE[SEQUENCE_ELEMENT]:
        experience = next(self.generator), 0.   # type: EXPERIENCE[str]
        return experience


class Text(Sequence[str]):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.permissible_non_letter = string.digits + string.punctuation + " "
        with open(self.file_path, mode="r") as file:
            self.text = file.read()
        self.text_len = len(self.text)
        self.index = -1

    def react_to(self, motor: Optional[NO_ACTION]) -> EXPERIENCE[SEQUENCE_ELEMENT]:
        while True:
            self.index = (self.index + 1) % self.text_len
            element = self.text[self.index]

            if element in string.ascii_letters:
                experience = element.lower(), 0.    # type: EXPERIENCE[str]
                return experience

            if element in self.permissible_non_letter:
                experience = element, 0.            # type: EXPERIENCE[str]
                return experience


class ExchangeRates(Sequence[float]):
    def __init__(self, file_path: str, seconds_interval: int = 60, start: int = -1, end: int = -1):
        self.generator = sequence_rational_crypto(file_path, seconds_interval, start_val=start, end_val=end)

    def react_to(self, motor: Optional[NO_ACTION]) -> EXPERIENCE[SEQUENCE_ELEMENT]:
        experience = next(self.generator), 0.   # type: EXPERIENCE[float]
        return experience


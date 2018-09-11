#!/usr/bin/env python3
# coding=utf-8
import string
from math import sin, cos
from typing import Tuple

from data_generation.systems.abstract_classes import Sequence, ELEMENT
from environments.non_interactive import sequence_nominal_alternating


class ArtificialNominal(Sequence[str]):
    def __init__(self, history_length: int = 1):
        super().__init__(history_length=history_length)
        self.generator = sequence_nominal_alternating()

    def _next_element(self) -> ELEMENT:
        return next(self.generator)


class Text(Sequence[str]):
    def __init__(self, file_path: str, history_length: int = 1):
        super().__init__(history_length=history_length)
        self.file_path = file_path
        self.permissible_non_letter = string.digits + string.punctuation + " "
        with open(self.file_path, mode="r") as file:
            self.text = file.read()
        self.text_len = len(self.text)
        self.index = -1

    def _next_element(self) -> ELEMENT:
        while True:
            self.index = (self.index + 1) % self.text_len
            element = self.text[self.index]

            if element in string.ascii_letters:
                return element.lower()

            if element in self.permissible_non_letter:
                return element


class ArtificialRational(Sequence[float]):
    def __init__(self, history_length: int = 1):
        super().__init__(history_length=history_length)
        self.i = -1
        self.history = []

    def get_example(self) -> Tuple[Tuple[ELEMENT, ...], ELEMENT]:
        for _ in range(self.history_length - len(self.history)):
            self.i += 1
            self.history.append(sin(self.i / 100.))

        input_value = tuple(self.history)
        target_value = float(cos(self.i / 100.) >= 0.) * 2. - 1.

        for _ in range(len(self.history) - self.history_length + 1):
            self.history.pop(0)

        return input_value, target_value

    def _next_element(self) -> ELEMENT:
        pass
